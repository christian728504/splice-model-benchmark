import pysam
import os
import numpy as np
import polars as pl
import itertools
import multiprocessing

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

class GetIntronJunctions:
    def __init__(
        self,
        bam_path,
        threads,
        max_mismatch_rate,
        min_mapq,
        chroms = [f'chr{x}' for x in range(1, 23)] + ['chrX', 'chrY'],
        strand = ['+', '-'],
    ):
        self.bam_path = bam_path
        if not os.path.exists(self.bam_path + '.bai'):
            pysam.index(self.bam_path)
        self.chroms = chroms
        self.strand = strand
        self.threads = threads
        self.max_mismatch_rate = max_mismatch_rate
        self.min_mapq = min_mapq
    
    def run(self):
        args = []
        for chrom, strand in itertools.product(self.chroms, self.strand):
            args.append((chrom, strand, self.bam_path, self.max_mismatch_rate, self.min_mapq))
        
        with multiprocessing.Pool(processes=self.threads) as pool:
            results = pool.starmap(_process_chromosome_strand, args)
           
        total_mapped_reads = 0 
        with pysam.AlignmentFile(self.bam_path, "rb") as samfile:
            stats = samfile.get_index_statistics()
            for stat in stats:
                total_mapped_reads += stat.mapped
            
        bed_file = {'chrom': [], 'start': [], 'end': [], 'strand': [], 'reads': []}
        for temp_dict in results:
            bed_file['chrom'].extend(temp_dict['chrom'])
            bed_file['start'].extend(temp_dict['start'])
            bed_file['end'].extend(temp_dict['end'])
            bed_file['strand'].extend(temp_dict['strand'])
            bed_file['reads'].extend(temp_dict['reads'])
        
        bed_file_df = pl.from_dict(bed_file)
        
        return bed_file_df