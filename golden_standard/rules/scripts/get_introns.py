from snakemake.script import snakemake

import os
import pysam
import polars as pl
import multiprocessing
import itertools

def _process_chromosome_strand(chrom, strand, bam_path, max_mismatch_rate, min_mapq):
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
            
            if strand == '+':
                if not read.is_forward:
                    continue
            else:
                if not read.is_reverse:
                    continue
                
            yield read
    
    with pysam.AlignmentFile(bam_path, "rb") as samfile:
        intron_dict = samfile.find_introns(_read_generator(samfile))
        temp_dict = {'chrom': [], 'start': [], 'end': [], 'strand': [], 'reads': []}
        
        for (start, end), reads in intron_dict.items():
            temp_dict['chrom'].append(chrom)
            temp_dict['start'].append(start)
            temp_dict['end'].append(end)
            temp_dict['strand'].append(strand)
            temp_dict['reads'].append(reads)
            
        return temp_dict

bam_path = snakemake.input.bam_file
threads = snakemake.threads
max_mismatch_rate = snakemake.params.max_mismatch_rate
min_mapq = snakemake.params.min_mapq
chroms = [f'chr{x}' for x in range(1, 23)] + ['chrX', 'chrY']
strand = ['+', '-']

if not os.path.exists(bam_path + '.bai'):
    pysam.index(bam_path)
args = []
for chrom, strand_val in itertools.product(chroms, strand):
    args.append((chrom, strand_val, bam_path, max_mismatch_rate, min_mapq))
with multiprocessing.Pool(processes=threads) as pool:
    results = pool.starmap(_process_chromosome_strand, args)

bed_file = {'chrom': [], 'start': [], 'end': [], 'strand': [], 'reads': []}
for temp_dict in results:
    bed_file['chrom'].extend(temp_dict['chrom'])
    bed_file['start'].extend(temp_dict['start'])
    bed_file['end'].extend(temp_dict['end'])
    bed_file['strand'].extend(temp_dict['strand'])
    bed_file['reads'].extend(temp_dict['reads'])

bed_file_df = pl.from_dict(bed_file)
bed_file_df.write_csv(snakemake.output.bed_file, separator='\t')
