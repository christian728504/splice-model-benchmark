import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"

import zarr
import jax
import numpy as np
import sys
from collections import Counter
import polars as pl
import haiku as hk
import jax.numpy as jnp
from pyfaidx import Fasta
from nucleotide_transformer.pretrained import get_pretrained_segment_nt_model
from sklearn.metrics import precision_recall_curve, auc
from matplotlib import pyplot as plt
from Bio.Seq import reverse_complement
from tqdm import tqdm
from urllib.request import urlretrieve

class SegmentNTEvaluator:
    def __init__(self,
                 gencode_gtf: str = "reference_files/gencode.v29.primary_assembly.annotation_UCSC_names.gtf.parquet",
                 transcript_quantifications: tuple = ('reference_files/transcript_quantifications_rep1.tsv', 'reference_files/transcript_quantifications_rep2.tsv'),
                 consensus_fasta: str = 'reference_files/GM12878.fasta',
                 sequence_length: int = 30000,
                 batch_size: int = 3,
                 transcript_count_threshold: int = 2,
                 filter_transcripts: bool = True,
                 predicitons_path: str = 'results/segmentnt_predictions.zarr',
                 splice_sites_path: str = 'reference_files/splice_sites.parquet',
                 aurpc_plot_path: str = 'results/segmentnt.svg',
                 stratified_auprc_plot_path: str = 'results/segmentnt_stratified_{type}.svg'):
        self.gtf_file = gencode_gtf
        self.consensus_fasta = consensus_fasta
        self.transcript_quantifications = transcript_quantifications
        if not os.path.exists(self.transcript_quantifications[0]):
            urlretrieve("https://www.encodeproject.org/files/ENCFF971DVB/@@download/ENCFF971DVB.tsv", self.transcript_quantifications[0])
        if not os.path.exists(self.transcript_quantifications[1]):
            urlretrieve("https://www.encodeproject.org/files/ENCFF189XTO/@@download/ENCFF189XTO.tsv", self.transcript_quantifications[1])
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.transcript_count_threshold = transcript_count_threshold
        self.filter_transcripts = filter_transcripts
        self.predictions_path = predicitons_path
        self.aurpc_plot_path = aurpc_plot_path
        self.stratified_auprc_plot_path = stratified_auprc_plot_path
        self._zarr_root = zarr.group(store=zarr.DirectoryStore(self.predictions_path))
        self.target_chromosomes = ['chr20', 'chr21']
        self._acceptor_predictions = self._zarr_root.require_group('acceptor_predictions')
        self._donor_predictions = self._zarr_root.require_group('donor_predictions')
        self._acceptor_truth = self._zarr_root.require_group('acceptor_truth')
        self._donor_truth = self._zarr_root.require_group('donor_truth')
        self.splice_sites_path = splice_sites_path
        self._splice_sites_df = pl.read_parquet(self.splice_sites_path)


    def filter_gencode(self):
        """Helper method to filter GENCODE GTF data."""
        print("Filtering GENCODE GTF...")
        fasta = Fasta(self.consensus_fasta)
        quant_tsv_1 = pl.read_csv(self.transcript_quantifications[0], separator='\t')
        quant_tsv_2 = pl.read_csv(self.transcript_quantifications[1], separator='\t')
        joined_tsv = quant_tsv_1.join(quant_tsv_2, on='transcript_ID', how='inner')
        averaged_counts = joined_tsv.with_columns(
            ((pl.col('rep1ENCSR368UNC') + pl.col('rep2ENCSR368UNC')) / 2).alias('transcript_count')
        )
        clean_tsv = averaged_counts.select("annot_transcript_id", "annot_transcript_name", "transcript_count")
        expressed_transcripts = clean_tsv.filter(pl.col('transcript_count') >= self.transcript_count_threshold)['annot_transcript_id'].to_list()
        print(f"Number of expressed transcripts: {len(expressed_transcripts)}")
        
        gtf = pl.read_parquet(self.gtf_file).drop('index')
        filtered_df = gtf.filter(
            (pl.col('feature') == 'exon') &
            (pl.col('gene_type') == 'protein_coding') &
            (pl.col('seqname').is_in(self.target_chromosomes))
        )
        if self.filter_transcripts:
            filtered_df = filtered_df.filter(pl.col('transcript_id').is_in(expressed_transcripts))
            
        transcript_counts = (
            filtered_df
            .select(['seqname', 'transcript_id'])
            .unique()
            .group_by('seqname')
            .count()
            .sort('seqname')
        )
        print(f"Number of transcripts per chromsome: {transcript_counts}")
        
        with_start = filtered_df.with_columns(
            pl.lit('start').alias('pos_type'),
            pl.col('start').alias('pos'))
        with_end = filtered_df.with_columns(
            pl.lit('end').alias('pos_type'),
            pl.col('end').alias('pos')
        )

        concatenated = pl.concat([with_start, with_end])
        drop_start_and_end = concatenated.drop('start', 'end')
        exon_number_as_int = drop_start_and_end.with_columns(pl.col('exon_number').cast(pl.Int64))
        sorted_df = exon_number_as_int.sort('seqname', 'transcript_id', 'exon_number', 'pos')

        grouped = sorted_df.group_by('seqname', 'transcript_id').agg(pl.col('pos'))
        remove_single_exons = grouped.filter(pl.col('pos').list.len() > 2)
        removed_start_and_end = remove_single_exons.with_columns(
            pl.col("pos").list.slice(1, pl.col("pos").list.len() - 2)
            .alias("pos")
        )

        exploded_df = removed_start_and_end.explode('pos')
        joined_df = sorted_df.join(exploded_df, on=['seqname', 'pos', 'transcript_id'], how='inner')
        unique_transcripts = joined_df.sort('seqname', 'pos', 'strand', 'gene_id', descending=True).unique(
            subset=['seqname', 'pos', 'strand', 'gene_id'], keep='first', maintain_order=True
        )
        
        seqname_to_chrom = unique_transcripts.with_columns(pl.col('seqname').alias('chrom').cast(pl.String)).drop('seqname')
        
        def get_seq(row):
            chrom, pos, strand, pos_type = row["chrom"], row["pos"], row["strand"], row["pos_type"]
            if strand == "+":
                if pos_type == "start":
                    return fasta[chrom][pos-3:pos-1].seq
                else:
                    return fasta[chrom][pos:pos+2].seq
            else:
                if pos_type == "start":
                    return reverse_complement(fasta[chrom][pos-3:pos-1].seq)
                else:
                    return reverse_complement(fasta[chrom][pos:pos+2].seq)
                
        added_sequence = seqname_to_chrom.with_columns([
            pl.when((pl.col("strand") == "+") & (pl.col("pos_type") == "start")).then(pl.lit("acceptor"))
                .when((pl.col("strand") == "-") & (pl.col("pos_type") == "end")).then(pl.lit("acceptor"))
                .otherwise(pl.lit("donor")).alias("type"),
            pl.struct(["chrom", "pos", "strand", "pos_type"]).map_elements(get_seq, return_dtype=pl.Utf8).alias("sequence")
        ]).with_row_index()
        
        added_sequence.write_parquet(self.splice_sites_path)
        self._splice_sites_df = pl.read_parquet(self.splice_sites_path)
    
    
    def get_ground_truth(self):
        """Generate ground truth binary arrays for each chromosome."""
        fasta = Fasta(self.consensus_fasta)

        for chrom in tqdm(self.target_chromosomes, desc="Generating ground truth"):
            donor_sites = np.zeros(len(fasta[chrom]), dtype=np.uint8)
            acceptor_sites = np.zeros(len(fasta[chrom]), dtype=np.uint8)
            chrom_df = self._splice_sites_df.filter(pl.col('chrom') == chrom)
            
            [donor_indices] = chrom_df.filter(pl.col('type') == 'donor').group_by('chrom').agg(pl.col('pos'))['pos'].to_list()
            donor_incides_np = np.array(donor_indices) - 1
            donor_sites[donor_incides_np] = 1
            
            [acceptor_indices] = chrom_df.filter(pl.col('type') == 'acceptor').group_by('chrom').agg(pl.col('pos'))['pos'].to_list()
            acceptor_indices_np = np.array(acceptor_indices) - 1
            acceptor_sites[acceptor_indices_np] = 1
            
            if chrom not in self._acceptor_truth and chrom not in self._donor_truth:
                self._acceptor_truth.create_dataset(chrom, data=acceptor_sites, dtype=np.uint8)
                self._donor_truth.create_dataset(chrom, data=donor_sites, dtype=np.uint8)
            else:
                print(f"Chromosome {chrom} already exists in truth datasets")

            
    def _process_batch(self, batch, args):
        """Process a batch of sequences and update predictions for entire windows."""
        parameters, apply_fn, tokenizer, keys, donor_idx, acceptor_idx = args
        sequences = list(batch.values())
        tokens_ids = [b[1] for b in tokenizer.batch_tokenize(sequences)]
        tokens = jnp.asarray(tokens_ids, dtype=jnp.int32)
        outputs = apply_fn(parameters, keys, tokens)
        probabilities = jax.nn.softmax(outputs["logits"], axis=-1)[..., -1]
            
        for i, (chrom, index) in enumerate(batch.keys()):
            donor_window = np.array(probabilities[i, 12500:17500, donor_idx])
            acceptor_window = np.array(probabilities[i, 12500:17500, acceptor_idx])
            
            half_window = 5000 // 2
            window_start = index - half_window
            window_end = window_start + 5000
            
            acceptor_mask = self._acceptor_predictions[chrom][window_start:window_end] != 0
            donor_mask = self._donor_predictions[chrom][window_start:window_end] != 0
            current_acceptor = self._acceptor_predictions[chrom][window_start:window_end]
            current_donor = self._donor_predictions[chrom][window_start:window_end]
            
            alpha = 0.5
            
            if np.any(acceptor_mask):
                current_acceptor[acceptor_mask] = current_acceptor[acceptor_mask] * (1-alpha) + acceptor_window[acceptor_mask] * alpha
            if np.any(donor_mask):
                current_donor[donor_mask] = current_donor[donor_mask] * (1-alpha) + donor_window[donor_mask] * alpha
            current_acceptor[np.logical_not(acceptor_mask)] = acceptor_window[np.logical_not(acceptor_mask)]
            current_donor[np.logical_not(donor_mask)] = donor_window[np.logical_not(donor_mask)]
            
            self._acceptor_predictions[chrom][window_start:window_end] = current_acceptor
            self._donor_predictions[chrom][window_start:window_end] = current_donor
    
    
    def generate_segmentnt_predictions(self):
        """Generate SegmentNT predictions for all splice sites."""
        fasta = Fasta(self.consensus_fasta)
        for chrom in tqdm(self.target_chromosomes, desc="Creating datasets"):
            donor_sites = np.zeros(len(fasta[chrom]))
            acceptor_sites = np.zeros(len(fasta[chrom]))
            if chrom not in self._acceptor_predictions and chrom not in self._donor_predictions:
                self._acceptor_predictions.create_dataset(chrom, data=acceptor_sites, dtype=np.float64)
                self._donor_predictions.create_dataset(chrom, data=donor_sites, dtype=np.float64)
            else:
                print(f"Chromosome {chrom} already exists in prediction datasets")
        
        if self.sequence_length % 6 != 0:
            raise ValueError(f"Sequence length ({self.sequence_length}) must be divisible by 6 (nucleotides per token)")
        max_tokens = self.sequence_length // 6
        if max_tokens % 4 != 0:
            raise ValueError(f"Number of tokens ({max_tokens}) must be divisible by 4 due to model downsampling requirements")
        if max_tokens + 1 > 5001:
            inference_rescaling_factor = (max_tokens + 1) / 2048
        else:
            inference_rescaling_factor = None
            
        parameters, forward_fn, tokenizer, config = get_pretrained_segment_nt_model(
            model_name="segment_nt",
            rescaling_factor=inference_rescaling_factor,
            max_positions=max_tokens + 1,
        )
        forward_fn = hk.transform(forward_fn)
        
        device = jax.devices("gpu")[0]
        apply_fn = forward_fn.apply
        random_key = jax.random.PRNGKey(seed=0)
        keys = jax.device_put(random_key, device=device)
        parameters = jax.device_put(parameters, device=device)
        donor_idx = config.features.index('splice_donor')
        acceptor_idx = config.features.index('splice_acceptor')
            
        args = (parameters, apply_fn, tokenizer, keys, donor_idx, acceptor_idx)
        
        batch = {}
        
        for site in tqdm(self._splice_sites_df.iter_rows(named=True), total=len(self._splice_sites_df), desc="Processing sites"):
            chrom = site['chrom']
            index = site['pos'] - 1
            
            total_length = self.sequence_length
            half_total = total_length // 2
            
            window_start = max(0, index - half_total)
            window_end = min(len(fasta[chrom]), index + half_total)
            seq = str(fasta[chrom][window_start:window_end])
            
            batch_key = (chrom, index)
            
            batch[batch_key] = seq

            if len(batch) == self.batch_size:
                self._process_batch(batch, args)
                batch = {}
        
        if batch:
            self._process_batch(batch, args)
    

    def calculate_and_plot_metrics(self):
        """
        Calculate AUPRC and other metrics using only regions where predictions were made.
        """
        args = [(self._acceptor_predictions, self._acceptor_truth, "acceptor"), 
                (self._donor_predictions, self._donor_truth, "donor")]
        metadata = self._splice_sites['metadata'][:]
        
        results = {}
        
        for pred_dataset, truth_dataset, site_type in args:    
            type_metadata = metadata[metadata['type'] == site_type]
            window_size = 5000
            total_sites = len(type_metadata)
            total_size = total_sites * window_size
            
            ground_truth = np.zeros(total_size, dtype=np.int8)
            predictions = np.zeros(total_size, dtype=np.float64)

            current_idx = 0
            for site in tqdm(type_metadata, desc="Extracting prediciton and truth windows", miniters=1000):
                index = site['pos'] - 1
                chrom = site['chrom']
                
                half_window = 5000 // 2
                window_start = index - half_window
                window_end = window_start + 5000
                
                site_prediction = pred_dataset[chrom][window_start:window_end]
                site_truth = truth_dataset[chrom][window_start:window_end]
                
                predictions[current_idx:current_idx+5000] = site_prediction
                ground_truth[current_idx:current_idx+5000] = site_truth
                current_idx += 5000
            
            precision, recall, _ = precision_recall_curve(ground_truth, predictions)
            auprc = auc(recall, precision)
            
            k = int(np.sum(ground_truth))
            top_k_indices = np.argsort(predictions)[-k:]
            top_k_accuracy = np.sum(ground_truth[top_k_indices]) / k
            
            results[site_type] = {
                'precision': precision,
                'recall': recall,
                'auprc': auprc,
                'topk': top_k_accuracy
            }
        
        mean_auprc = (results['acceptor']['auprc'] + results['donor']['auprc']) / 2
        mean_topk = (results['acceptor']['topk'] + results['donor']['topk']) / 2
        
        plt.figure(figsize=(10, 6))
        plt.plot(results['acceptor']['recall'], results['acceptor']['precision'], 
                label=f'Acceptor (AUPRC={results["acceptor"]["auprc"]:.3f})')
        plt.plot(results['donor']['recall'], results['donor']['precision'], 
                label=f'Donor (AUPRC={results["donor"]["auprc"]:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curves\nMean AUPRC: {mean_auprc:.3f}, Mean Top-k: {mean_topk:.3f}')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.aurpc_plot_path, dpi=300, format='svg')
            
            
    def calculate_and_plot_metrics_stratified(self):
        """
        Calculate AUPRC and top-k accuracy stratified by the splice site recognition sequence.
        """
        args = [(self._acceptor_predictions, self._acceptor_truth, "acceptor"), 
                        (self._donor_predictions, self._donor_truth, "donor")]

        results = {}

        for pred_dataset, truth_dataset, site_type in args:    
            type_df = self._splice_sites_df.filter(pl.col('type') == site_type)
            
            sequence_counts = Counter(type_df['sequence'])
            top_sequences = [seq for seq, _ in sequence_counts.most_common(10)]
            
            results[site_type] = {}
            
            for sequence in top_sequences:
                seq_df = type_df.filter(pl.col('sequence') == sequence)
                
                window_size = 5000
                total_sites = len(seq_df)
                total_size = total_sites * window_size
                
                ground_truth = np.zeros(total_size, dtype=np.int8)
                predictions = np.zeros(total_size, dtype=np.float64)
                
                current_idx = 0
                for site in seq_df.iter_rows(named=True):
                    index = site['pos'] - 1
                    chrom = site['chrom']
                    
                    half_window = window_size // 2
                    window_start = index - half_window
                    window_end = window_start + window_size
                    
                    site_prediction = pred_dataset[chrom][window_start:window_end]
                    site_truth = truth_dataset[chrom][window_start:window_end]
                    
                    predictions[current_idx:current_idx+window_size] = site_prediction
                    ground_truth[current_idx:current_idx+window_size] = site_truth
                    current_idx += window_size
                
                precision, recall, _ = precision_recall_curve(ground_truth, predictions)
                auprc = auc(recall, precision)
                
                k = int(np.sum(ground_truth))
                top_k_indices = np.argsort(predictions)[-k:]
                top_k_accuracy = np.sum(ground_truth[top_k_indices]) / k
                
                results[site_type][sequence] = {
                    'precision': precision,
                    'recall': recall,
                    'auprc': auprc,
                    'topk': top_k_accuracy,
                    'count': total_sites
                }

        plt.figure(figsize=(12, 8))
        for sequence, data in results['acceptor'].items():
            plt.plot(data['recall'], data['precision'], 
                    label=f'{sequence} (AUPRC={data["auprc"]:.3f}, n={data["count"]}, topk={data["topk"]:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Acceptor Precision-Recall Curves by Sequence')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.stratified_auprc_plot_path.format(type='acceptor'), dpi=300, format='svg')

        plt.figure(figsize=(12, 8))
        for sequence, data in results['donor'].items():
            plt.plot(data['recall'], data['precision'], 
                    label=f'{sequence} (AUPRC={data["auprc"]:.3f}, n={data["count"]}), topk={data["topk"]:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Donor Precision-Recall Curves by Sequence')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.stratified_auprc_plot_path.format(type='donor'), dpi=300, format='svg')           
