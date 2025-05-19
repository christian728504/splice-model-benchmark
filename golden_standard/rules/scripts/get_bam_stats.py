from snakemake.script import snakemake

import subprocess
import polars as pl
import matplotlib.pyplot as plt

cmd = ["stats_from_bam", f"{snakemake.input.bam_file}", "-t", f"{snakemake.threads}"]
process = subprocess.run(
    cmd,
    check=False,
    capture_output=True,
    text=True
)
with open(snakemake.output.stats_file, 'w') as f:
    f.write(process.stdout)

stats_df = pl.read_csv(snakemake.output.stats_file, separator='\t')

plt.style.use('cowplot')
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig, axes = plt.subplots(2, 1, figsize=(10, 10))
fig.suptitle('BAM file Statistics (ENCFF219UJG.bam)')
axes[0].hist(stats_df['read_length'], bins=100, alpha=0.5, label='Read Length', color=color_cycle[0])
axes[0].set_xlabel('Read Length')
axes[0].set_ylabel('Count')
axes[1].hist(stats_df['acc'], bins=100, alpha=0.5, label='Read Accuracy', color=color_cycle[0])
axes[1].set_xlabel('Accuracy')
axes[1].set_ylabel('Count')
axes[1].ticklabel_format(style='plain', axis='y')
for axis in axes:
    y_ticks = axis.get_yticks()
    padding = y_ticks[1] * 0.1
    y_lim_min = y_ticks[0] - padding
    axis.set_ylim(bottom=y_lim_min)

plt.savefig(snakemake.output.plot)