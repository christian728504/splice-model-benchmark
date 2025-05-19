from pyfaidx import Fasta
import pysam
import os
import numpy as np
import polars as pl
import collections
import itertools
import multiprocessing

def _process_chromosome_strand(chrom, strand, bam_path, fasta_path):
    def _shannon_entropy(counts):
        if len(counts) <= 2:
            return np.float64(0)
        probs = counts / np.sum(counts)
        return -np.sum(probs * np.log2(probs))
    
    # short-read BAM files seem to have filters applied to them during alignment
    # using STAR, so the filtering in this generator may not be necessary. I'll
    # keep it around just incase.
    # def _read_generator(samfile, chrom, strand):
    #     for read in samfile.fetch(chrom, multiple_iterators=True):
    #         if read.is_unmapped:
    #             continue
    #         if read.is_supplementary:
    #             continue
    #         if read.is_duplicate:
    #             continue
    #         if read.is_qcfail:
    #             continue
    #         if read.mapping_quality < 20:
    #             continue
    #         if strand == '+':
    #             if not read.is_forward:
    #                 continue
    #         else:
    #             if not read.is_reverse:
    #                 continue
    #         yield read
    
    def _read_generator(samfile, chrom, strand):
        for read in samfile.fetch(chrom, multiple_iterators=True):
            if strand == '+':
                if not read.is_forward:
                    continue
            else:
                if not read.is_reverse:
                    continue
            yield read
    
    with pysam.AlignmentFile(bam_path, "rb") as samfile, Fasta(fasta_path) as fasta:
        intron_dict = samfile.find_introns(_read_generator(samfile, chrom, strand))
        temp_dict = {'chrom': [], 'start': [], 'end': [], 'strand': [], 'reads': [], 'sequence': [], 'shannon_entropy': []}
        
        junction_to_blocks = collections.defaultdict(list)
        for read in _read_generator(samfile, chrom, strand):
            blocks = read.get_blocks()

            if len(blocks) < 2:
                continue

            for i in range(len(blocks) - 1):
                junction_start = blocks[i][1]
                junction_end = blocks[i+1][0]

                if (junction_start, junction_end) in intron_dict.keys():
                    first_block_start = blocks[i][0]
                    first_block_end = blocks[i][1]

                    junction_to_blocks[(junction_start, junction_end)].append(first_block_end - first_block_start)
        
        junction_entropy = {}
        for junction, block_lengths in junction_to_blocks.items():
            counter_dict = collections.Counter(block_lengths)
            counts = np.array(list(counter_dict.values()))
            junction_entropy[junction] = _shannon_entropy(counts)
            
        for (start, end), reads in intron_dict.items():
            temp_dict['chrom'].append(chrom)
            temp_dict['start'].append(start)
            temp_dict['end'].append(end)
            temp_dict['strand'].append(strand)
            temp_dict['reads'].append(reads)
            temp_dict['sequence'].append(fasta[chrom][start:end].seq)
            temp_dict['shannon_entropy'].append(junction_entropy.get((start, end), np.float64(-1)))
            
        return temp_dict

class IntronJunctions:
    def __init__(
        self, bam_path, 
        fasta_path = 'GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta',
        chroms = [f'chr{x}' for x in range(1, 23)] + ['chrX', 'chrY'],
        strand = ['+', '-']
    ):
        self.bam_path = bam_path
        if not os.path.exists(self.bam_path + '.bai'):
            pysam.index(self.bam_path)
        self.fasta_path = fasta_path
        self.chroms = chroms
        self.strand = strand
    
    def get_intron_junctions(self):
        args = []
        for chrom, strand in itertools.product(self.chroms, self.strand):
            args.append((chrom, strand, self.bam_path, self.fasta_path))
        
        with multiprocessing.Pool(processes=32) as pool:
            results = pool.starmap(_process_chromosome_strand, args)
            
        bed_file = {'chrom': [], 'start': [], 'end': [], 'strand': [], 'reads': [], 'sequence': [], 'shannon_entropy': []}
        for temp_dict in results:
            bed_file['chrom'].extend(temp_dict['chrom'])
            bed_file['start'].extend(temp_dict['start'])
            bed_file['end'].extend(temp_dict['end'])
            bed_file['strand'].extend(temp_dict['strand'])
            bed_file['reads'].extend(temp_dict['reads'])
            bed_file['sequence'].extend(temp_dict['sequence'])
            bed_file['shannon_entropy'].extend(temp_dict['shannon_entropy'])
        
        return pl.from_dict(bed_file)