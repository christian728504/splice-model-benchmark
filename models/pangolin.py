import os
import zarr
import numpy as np
import polars as pl
from pyfaidx import Fasta
from sklearn.metrics import precision_recall_curve, auc
from pangolin.model import *
from matplotlib import pyplot as plt
from tqdm import tqdm
from Bio.Seq import Seq
from urllib.request import urlretrieve

from utils import one_hot_encode

class PangolinEvaluator:
    def __init__(self,
                 gencode_gtf: str = "reference_files/gencode.v29.primary_assembly.annotation_UCSC_names.gtf.parquet",
                 transcript_quantifications: tuple = ('reference_files/transcript_quantifications_rep1.tsv', 'reference_files/transcript_quantifications_rep2.tsv'),
                 consensus_fasta: str = 'reference_files/GM12878.fasta',
                 model_weights = "reference_files/pangolin/models/final.{model_index}.{model_num}.3",
                 sequence_length: int = 5000,
                 context_length: int = 10000,
                 batch_size: int = 128,
                 transcript_count_threshold: int = 2,
                 filter_transcripts: bool = True,
                 predicitons_path: str = 'results/pangolin_predictions.zarr',
                 aurpc_plot_path: str = 'results/pangolin.svg'):
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
        self._splice_site_predictions = self._zarr_root.require_group('splice_site_predictions')
        self._splice_site_truth = self._zarr_root.require_group('splice_site_truth')
        

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
            splice_sites = np.zeros(len(fasta[chrom]), dtype=np.uint8)
            chrom_df = sorted_df.filter(pl.col('seqname') == chrom)
            
            for row in chrom_df.iter_rows(named=True):
                if row['strand'] == '+':
                    if row['start'] != "EXCLUDE":
                        all_splice_site_data.append((chrom, int(row['start']) - 1, '+'))
                        splice_sites[int(row['start']) - 1] = 1
                    if row['end'] != "EXCLUDE":
                        all_splice_site_data.append((chrom, int(row['end']) - 1, '+'))
                        splice_sites[int(row['end']) - 1] = 1
                else:
                    if row['start'] != "EXCLUDE":
                        all_splice_site_data.append((chrom, int(row['start']) - 1, '-'))
                        splice_sites[int(row['start']) - 1] = 1
                    if row['end'] != "EXCLUDE":
                        all_splice_site_data.append((chrom, int(row['end']) - 1, '-'))
                        splice_sites[int(row['end']) - 1] = 1
            
            if chrom not in self._splice_site_truth:
                self._splice_site_truth.create_dataset(chrom, data=splice_sites, dtype=np.uint8)
            else:
                print(f"Chromosome {chrom} already exists in truth dataset")
        
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
            
            
    # adapted from https://github.com/tkzeng/Pangolin/blob/main/scripts/custom_usage.py
    def _process_batch(self, batch, models):
        """Process a batch of sequences and update predictions for entire windows."""
        INDEX_MAP = {0:1, 1:2, 2:4, 3:5, 4:7, 5:8, 6:10, 7:11}
        model_nums = [0, 2, 4, 6]
        
        sequences = list(batch.values())
        encoded_seqs = one_hot_encode(np.array(sequences))
        transposed_seqs = np.transpose(encoded_seqs, (0, 2, 1))
        X = torch.from_numpy(np.stack(transposed_seqs)).float().cuda()
        
        model_type_predictions = {model_num: [] for model_num in model_nums}
        
        for i, model_num in enumerate(model_nums):
            model_group = models[i*5:(i+1)*5]
            
            group_predictions = []
            for model in model_group:
                with torch.no_grad():
                    preds = model(X)
                    tissue_preds = preds[:, INDEX_MAP[model_num], :].cpu().numpy()
                    group_predictions.append(tissue_preds)
            
            model_type_predictions[model_num] = np.mean(group_predictions, axis=0)
        
        for i, (chrom, index, strand) in enumerate(batch.keys()):
            splice_site_window = np.mean([model_type_predictions[model_num][i, :] for model_num in model_nums], axis=0)
            
            if strand == '-':
                splice_site_window = np.flip(splice_site_window)
            
            half_window = self.sequence_length // 2
            window_start = index - half_window
            window_end = window_start + self.sequence_length
            
            splice_site_mask = self._splice_site_predictions[chrom][window_start:window_end] != 0
            current_splice_site = self._splice_site_predictions[chrom][window_start:window_end]
            
            alpha = 0.5
            if np.any(splice_site_mask):
                current_splice_site[splice_site_mask] = current_splice_site[splice_site_mask] * (1-alpha) + splice_site_window[splice_site_mask] * alpha
            current_splice_site[np.logical_not(splice_site_mask)] = splice_site_window[np.logical_not(splice_site_mask)]
            
            self._splice_site_predictions[chrom][window_start:window_end] = current_splice_site
                    

    def generate_pangolin_predictions(self):
        """Generate Pangolin predictions for all splice sites."""
        fasta = Fasta(self.consensus_fasta)

        model_nums = [0, 2, 4, 6]
        models = []
        for i in model_nums:
            for j in range(1, 6):
                model = Pangolin(L, W, AR)
                model.cuda()
                weights = torch.load(self.model_weights.format(model_index=j, model_num=i))
                model.load_state_dict(weights)
                model.eval()
                models.append(model)
                
        for chrom in tqdm(self.target_chromosomes, desc="Creating datasets"):
            splice_sites = np.zeros(len(fasta[chrom]))
            if chrom not in self._splice_site_predictions:
                self._splice_site_predictions.create_dataset(chrom, data=splice_sites, dtype=np.float64)
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
        metadata = self._splice_sites['metadata'][:]
        window_size = self.sequence_length
        total_sites = len(metadata)
        total_size = total_sites * window_size
        
        ground_truth = np.zeros(total_size, dtype=np.int8)
        predictions = np.zeros(total_size, dtype=np.float64)
        
        current_idx = 0
        for site in tqdm(metadata, desc="Extracting prediciton and truth windows", miniters=1000):
            index = site['index']
            chrom = site['chrom']
            
            half_window = self.sequence_length // 2
            window_start = index - half_window
            window_end = window_start + self.sequence_length
            
            site_prediction = self._splice_site_predictions[chrom][window_start:window_end]
            site_truth = self._splice_site_truth[chrom][window_start:window_end]
            
            predictions[current_idx:current_idx+self.sequence_length] = site_prediction
            ground_truth[current_idx:current_idx+self.sequence_length] = site_truth
            current_idx += self.sequence_length
            
        print("Calculating AUPRC and top-k accuracy...")
        precision, recall, _ = precision_recall_curve(ground_truth, predictions)
        auprc = auc(recall, precision)

        k = int(np.sum(ground_truth))
        top_k_indices = np.argsort(predictions)[-k:]
        topk = np.sum(ground_truth[top_k_indices]) / k

        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curves\nAURPC: {auprc:.3f}, Top-k: {topk:.3f}')
        plt.grid(True)
        plt.savefig(self.aurpc_plot_path, dpi=300, format='svg')

        print(f"AUPRC: {auprc:.4f}, Top-k: {topk:.4f}")
            
