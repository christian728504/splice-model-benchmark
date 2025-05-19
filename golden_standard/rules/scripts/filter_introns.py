from snakemake.script import snakemake

import polars as pl
import bioframe as bf
import warnings
warnings.filterwarnings("ignore", category=pl.exceptions.CategoricalRemappingWarning)

intron_df = pl.read_csv(snakemake.input.bed_file, separator='\t')
slim_gtf = pl.read_parquet(snakemake.params.gtf_parquet_path).select(['seqname', 'start', 'end', 'strand', 'feature', 'gene_id', 'gene_name', 'gene_type']).with_columns((pl.col('start') - 1).alias('start'), (pl.col('end') - 1).alias('end'))
filtered_gtf = slim_gtf.filter(pl.col('feature') == 'gene', pl.col('gene_type') == 'protein_coding').rename({'seqname': 'chrom'})
overlaps = bf.overlap(
    filtered_gtf.to_pandas(),
    intron_df.to_pandas(),
    on=['strand'],
    how='inner',
    suffixes=('_gene', '_intron')
)

# TODO: Get annotation for introns

contained = overlaps[
    (overlaps['start_intron'] >= overlaps['start_gene']) & 
    (overlaps['end_intron'] <= overlaps['end_gene'])
]
contained_pl = pl.from_pandas(contained)
inclusive_introns = contained_pl.unique(subset=['chrom_intron', 'start_intron', 'end_intron', 'strand_intron', 'gene_name_gene', 'reads_intron']).rename({'chrom_intron': 'chrom', 'start_intron': 'start', 'end_intron': 'end', 'strand_intron': 'strand', 'reads_intron': 'reads', 'gene_name_gene': 'gene_name'}).select('chrom', 'start', 'end', 'strand', 'gene_name', 'reads').sort('chrom', 'start')
exclusive_introns = intron_df.join(inclusive_introns, on=['chrom', 'start', 'end', 'strand', 'reads'], how='anti')
inclusive_introns.write_csv(snakemake.output.inclusive_introns_bed, separator='\t')
exclusive_introns.write_csv(snakemake.output.exclusive_introns_bed, separator='\t')