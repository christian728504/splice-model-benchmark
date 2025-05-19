from snakemake.script import snakemake

import os
os.environ['MPLBACKEND'] = 'Agg'
from pybedtools import BedTool
import subprocess

cmd = [
    "bamCoverage", "--bam", snakemake.input.bam_file, 
    "--outFileName", snakemake.output.forward_coverage_bed_file,
    "--outFileFormat", "bedgraph",
    "--samFlagExclude", "20",  # Exclude reads that map to the reverse strand and unmapped reads
    "--binSize", "1", 
    "--normalizeUsing", "RPKM",
    "--numberOfProcessors", str(snakemake.threads)
]
subprocess.run(cmd, check=True)