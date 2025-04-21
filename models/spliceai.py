import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "0"

import zarr
import numpy as np
import polars as pl
from pyfaidx import Fasta
from sklearn.metrics import precision_recall_curve, auc
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm
from Bio.Seq import Seq
from urllib.request import urlretrieve

from utils import one_hot_encode

class SpliceAIEvaluator:
    def __init__(self,
                 gencode_gtf: str = "reference_files/gencode.v29.primary_assembly.annotation_UCSC_names.gtf.parquet",
                 transcript_quantifications: tuple = ('reference_files/transcript_quantifications_rep1.tsv', 'reference_files/transcript_quantifications_rep2.tsv'),
                 consensus_fasta: str = 'reference_files/GM12878.fasta',
                 model_weights: str = "reference_files/spliceai/models/spliceai{index}.h5",
                 sequence_length: int = 5000,
                 context_length: int = 10000,
                 batch_size: int = 128,
                 transcript_count_threshold: int = 2,
                 filter_transcripts: bool = True,
                 predicitons_path: str = 'results/spliceai_predictions.zarr',
                 aurpc_plot_path: str = 'results/spliceai.png'):
        self.gtf_file = gencode_gtf
        self.consensus_fasta = consensus_fasta
        self.transcript_quantifications = transcript_quantifications
        if not os.path.exists(self.transcript_quantifications[0]):
            urlretrieve("https://www.encodeproject.org/files/ENCFF971DVB/@@download/ENCFF971DVB.tsv", self.transcript_quantifications[0])
        if not os.path.exists(self.transcript_quantifications[1]):
            urlretrieve("https://www.encodeproject.org/files/ENCFF189XTO/@@download/ENCFF189XTO.tsv", self.transcript_quantifications[1])
        self.model_weights = model_weights       
        self.sequence_length = sequence_length
        self.context_length = context_length
        self.batch_size = batch_size
        self.transcript_count_threshold = transcript_count_threshold
        self.filter_transcripts = filter_transcripts
        self.predictions_path = predicitons_path
        self.aurpc_plot_path = aurpc_plot_path
        self._zarr_root = zarr.group(store=zarr.DirectoryStore(self.predictions_path))
        self.target_chromosomes = ['chr1', 'chr3', 'chr5', 'chr7', 'chr9']
        self._splice_sites = self._zarr_root.require_group('splice_sites')
        self._acceptor_predictions = self._zarr_root.require_group('acceptor_predictions')
        self._donor_predictions = self._zarr_root.require_group('donor_predictions')
        self._acceptor_truth = self._zarr_root.require_group('acceptor_truth')
        self._donor_truth = self._zarr_root.require_group('donor_truth')


    def _filter_gencode(self):
        """Helper method to filter GENCODE GTF data."""
        print("Filtering GENCODE GTF...")
        quant_tsv_1 = pl.read_csv(self.transcript_quantifications[0], separator='\t')
        quant_tsv_2 = pl.read_csv(self.transcript_quantifications[1], separator='\t')
        joined_tsv = quant_tsv_1.join(quant_tsv_2, on='transcript_ID', how='inner')
        averaged_counts = joined_tsv.with_columns(
            ((pl.col('rep1ENCSR368UNC') + pl.col('rep2ENCSR368UNC')) / 2).alias('transcript_count')
        )
        clean_tsv = averaged_counts.select("annot_transcript_id", "annot_transcript_name", "transcript_count")
        expressed_transcripts = clean_tsv.filter(pl.col('transcript_count') >= self.transcript_count_threshold)['annot_transcript_id'].to_list()
        print(f"Number of expressed transcripts: {len(expressed_transcripts)}")
        
        gtf = pl.read_parquet(self.gtf_file)
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
        
        as_string = filtered_df.with_columns(pl.col('start').cast(pl.Utf8), pl.col('end').cast(pl.Utf8))
        as_num = as_string.with_columns(pl.col('exon_number').cast(pl.Int64))
        indexed_df = as_num.with_row_index()
        
        first_indices = []
        last_indices = []

        for _, group in indexed_df.group_by('transcript_id'):
            sorted_group = group.sort('exon_number')
            first_indices.append(sorted_group.row(0, named=True)['index'])
            last_indices.append(sorted_group.row(-1, named=True)['index'])

        # Create update expressions
        placeholder_df = indexed_df.with_columns([
            pl.when(pl.col("index").is_in(first_indices))
            .then(pl.lit("EXCLUDE"))
            .otherwise(pl.col("start"))
            .alias("start"),
            
            pl.when(pl.col("index").is_in(last_indices))
            .then(pl.lit("EXCLUDE"))
            .otherwise(pl.col("end"))
            .alias("end")
        ])

        sorted_df = placeholder_df.sort('seqname', 'transcript_id', 'exon_number')
        print(f"Done")
        
        return sorted_df
    
    
    def get_ground_truth(self):
        """Generate ground truth binary arrays and metadata in a single Zarr dataset."""
        print("Generating ground truth...")
        fasta = Fasta(self.consensus_fasta)
        sorted_df = self._filter_gencode()

        all_splice_site_data = []

        for chrom in self.target_chromosomes:
            donor_sites = np.zeros(len(fasta[chrom]), dtype=np.uint8)
            acceptor_sites = np.zeros(len(fasta[chrom]), dtype=np.uint8)
            chrom_df = sorted_df.filter(pl.col('seqname') == chrom)

            for row in chrom_df.iter_rows(named=True):
                if row['strand'] == '+':
                    if row['start'] != "EXCLUDE":
                        all_splice_site_data.append((chrom, int(row['start']) - 1, '+'))
                        acceptor_sites[int(row['start']) - 1] = 1
                    if row['end'] != "EXCLUDE":
                        all_splice_site_data.append((chrom, int(row['end']) - 1, '+'))
                        donor_sites[int(row['end']) - 1] = 1
                else:
                    if row['start'] != "EXCLUDE":
                        all_splice_site_data.append((chrom, int(row['start']) - 1, '-'))
                        donor_sites[int(row['start']) - 1] = 1
                    if row['end'] != "EXCLUDE":
                        all_splice_site_data.append((chrom, int(row['end']) - 1, '-'))
                        acceptor_sites[int(row['end']) - 1] = 1
                        
            if chrom not in self._acceptor_truth and chrom not in self._donor_truth:
                self._acceptor_truth.create_dataset(chrom, data=acceptor_sites, dtype=np.uint8)
                self._donor_truth.create_dataset(chrom, data=donor_sites, dtype=np.uint8)
            else:
                print(f"Chromosome {chrom} already exists in truth datasets")

        # Create structured array for splice site metadata
        splice_site_dtype = np.dtype([
            ('chrom', 'U10'),
            ('index', np.int64),
            ('strand', 'U1')
        ])
        splice_site_array = np.array(all_splice_site_data, dtype=splice_site_dtype)

        if 'metadata' not in self._splice_sites:
            self._splice_sites.create_dataset('metadata', data=splice_site_array)
        else:
            print("Splice site metadata array already exists in dataset")

        print("Done")
            
    
    def _process_batch(self, batch, models):
        """Process a batch of sequences and update predictions for entire windows."""
        sequences = list(batch.values())
        X = one_hot_encode(np.array(sequences))
        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        
        predictions = []
        for model in models:
            pred = model(X_tensor, training=False)
            if isinstance(pred, np.ndarray):
                predictions.append(pred)
            else:
                predictions.append(pred.numpy())
        
        avg_preds = np.mean(predictions, axis=0)  # Shape: (batch_size, sequence_length, 3)
        
        for i, (chrom, index, strand) in enumerate(batch.keys()):
            half_total = self.sequence_length // 2
            window_start = index - half_total
            relative_pos = index - window_start
            
            if strand == '-':
                center_pos = self.sequence_length - 1 - relative_pos
            else:
                center_pos = relative_pos
            
            acceptor_pred = avg_preds[i, center_pos, 1]  # 1: Acceptor
            donor_pred = avg_preds[i, center_pos, 2]     # 2: Donor
            
            self._acceptor_predictions[chrom][index] = acceptor_pred
            self._donor_predictions[chrom][index] = donor_pred
                        

    def generate_spliceai_predictions(self):
        """Generate SpliceAI predictions for all splice sites."""
        fasta = Fasta(self.consensus_fasta)
        models = []
        for idx in range(1, 6):
            model = tf.keras.models.load_model(self.model_weights.format(index=idx), compile=False)
            models.append(model)
        
        for chrom in tqdm(self.target_chromosomes, desc="Creating datasets"):
            donor_sites = np.zeros(len(fasta[chrom]))
            acceptor_sites = np.zeros(len(fasta[chrom]))
            if chrom not in self._acceptor_predictions and chrom not in self._donor_predictions:
                self._acceptor_predictions.create_dataset(chrom, data=acceptor_sites, dtype=np.float64)
                self._donor_predictions.create_dataset(chrom, data=donor_sites, dtype=np.float64)
            else:
                print(f"Chromosome {chrom} already exists in prediction datasets")
        
        metadata = self._splice_sites['metadata'][:]
        
        batch = {}
        for site in tqdm(metadata, desc="Processing sites"):
            chrom = site['chrom']
            index = site['index']
            strand = site['strand']
            
            total_length = self.sequence_length + self.context_length
            half_total = total_length // 2
            
            window_start = max(0, index - half_total)
            window_end = min(len(fasta[chrom]), index + half_total)
            seq = str(fasta[chrom][window_start:window_end])
            
            if strand == '-':
                seq = str(Seq(seq).reverse_complement())
            
            batch_key = (chrom, index, strand)
            
            batch[batch_key] = seq

            if len(batch) == self.batch_size:
                self._process_batch(batch, models)
                batch = {}
        
        if batch:
            self._process_batch(batch, models)
    

    def calculate_and_plot_metrics(self):
        """
        Calculate AUPRC and other metrics using only regions where predictions were made.
        """
        args = [(self._acceptor_predictions, self._acceptor_truth, "acceptor"), (self._donor_predictions, self._donor_truth, "donor")]
        
        for arg in args:
            masked_truth = []
            masked_preds = []
            for chrom in self.target_chromosomes:
                chrom_predictions = np.array(arg[0][chrom])
                chrom_ground_truth = np.array(arg[1][chrom])
                predictions_mask = chrom_predictions != 0
                masked_truth.append(chrom_ground_truth[predictions_mask])
                masked_preds.append(chrom_predictions[predictions_mask])

            ground_truth = np.concatenate(masked_truth)
            predictions = np.concatenate(masked_preds)
            
            precision, recall, _ = precision_recall_curve(ground_truth, predictions)
            auprc = auc(recall, precision)

            k = int(np.sum(ground_truth))
            top_k_indices = np.argsort(predictions)[-k:]
            top_k_accuracy = np.sum(ground_truth[top_k_indices]) / k
            
            if arg[2] == "acceptor":
                acc_precision, acc_recall, acc_auprc, acc_topk = precision, recall, auprc, top_k_accuracy
            else:
                don_precision, don_recall, don_auprc, don_topk = precision, recall, auprc, top_k_accuracy
        
        mean_auprc = (acc_auprc + don_auprc) / 2
        mean_topk = (acc_topk + don_topk) / 2

        plt.figure(figsize=(10, 6))
        plt.plot(acc_recall, acc_precision, label=f'Acceptor (AUPRC={acc_auprc:.3f})')
        plt.plot(don_recall, don_precision, label=f'Donor (AUPRC={don_auprc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curves\nMean AUPRC: {mean_auprc:.3f}, Mean Top-k: {mean_topk:.3f}')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.aurpc_plot_path, dpi=300)

        print(f"Acceptor AUPRC: {acc_auprc:.4f}, Top-k: {acc_topk:.4f}")
        print(f"Donor AUPRC: {don_auprc:.4f}, Top-k: {don_topk:.4f}")
        print(f"Mean AUPRC: {mean_auprc:.4f}, Mean Top-k: {mean_topk:.4f}")
            
