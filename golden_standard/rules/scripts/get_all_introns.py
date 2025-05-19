from snakemake.script import snakemake

import os
import pysam
import polars as pl
import multiprocessing
from tqdm import tqdm
import itertools

def _process_chromosome_strand(chrom, strand, bam_path, max_mismatch_rate, min_mapq, library_strand_specificity):
    """Process a single chromosome and strand from a BAM file and return dictionary of junction tuples to read counts"""
    def _read_generator(samfile):
        for read in samfile.fetch(chrom, multiple_iterators=True):
            if read.is_unmapped or \
                read.is_secondary or \
                read.is_supplementary or \
                read.is_qcfail or \
                read.mapping_quality < min_mapq:
                continue
            
            if read.has_tag('NM'):
                nm = read.get_tag('NM')
                read_length = read.query_length
                mismatch_rate = nm / read_length
                if mismatch_rate > max_mismatch_rate:
                    continue
            else:
                continue
            
            is_reverse = (read.flag & 0x10) != 0
            
            if library_strand_specificity.lower() == 'unstranded':
                if strand == '+' and is_reverse:
                    continue
                elif strand == '-' and not is_reverse:
                    continue
            elif library_strand_specificity.lower() == 'forward':
                if strand == '+' and not is_reverse:  
                    continue
                elif strand == '-' and is_reverse:
                    continue
            elif library_strand_specificity.lower() == 'reverse':
                if strand == '+' and is_reverse:
                    continue
                elif strand == '-' and not is_reverse:
                    continue
            
            yield read
    
    with pysam.AlignmentFile(bam_path, "rb") as samfile:
        basic_intron_dict = samfile.find_introns(_read_generator(samfile))
        
        junction_dict = {}
        for (start, end), reads in basic_intron_dict.items():
            junction_key = (chrom, start, end, strand)
            junction_dict[junction_key] = reads
            
        return junction_dict

def main():
    bam_files = snakemake.input.bam_files
    threads = snakemake.threads
    max_mismatch_rate = snakemake.params.max_mismatch_rate
    min_mapq = snakemake.params.min_mapq
    metadata = pl.read_csv(snakemake.params.metadata_file, separator='\t')
    chroms = [f'chr{x}' for x in range(1, 23)] + ['chrX', 'chrY']
    strands = ['+', '-']
    
    args = []
    for bam_path in bam_files:
        file_accession = os.path.basename(bam_path).split('.')[0]
        library_strand_specific = metadata.filter(pl.col('File accession') == file_accession)['Library strand-specific'].item()
        for chrom, strand in itertools.product(chroms, strands):
            args.append((chrom, strand, bam_path, max_mismatch_rate, min_mapq, library_strand_specific))
    
    print(f"Processing {len(bam_files)} BAM files across {len(chroms)} chromosomes and {len(strands)} strands")
    total = len(args)
    with multiprocessing.Pool(processes=threads) as pool:
        results = pool.starmap(_process_chromosome_strand, tqdm(args, total=total))
    
    aggregated_junctions = {}
    for junction_dict in results:
        for junction_key, reads in junction_dict.items():
            aggregated_junctions[junction_key] = aggregated_junctions.get(junction_key, 0) + reads
    
    final_bed = {'chrom': [], 'start': [], 'end': [], 'strand': [], 'reads': []}
    for (chrom, start, end, strand), reads in aggregated_junctions.items():
        final_bed['chrom'].append(chrom)
        final_bed['start'].append(start)
        final_bed['end'].append(end)
        final_bed['strand'].append(strand)
        final_bed['reads'].append(reads)
    
    bed_file_df = pl.from_dict(final_bed)
    bed_file_df.write_csv(snakemake.output.introns, separator='\t')
    
if __name__ == "__main__":
    main()