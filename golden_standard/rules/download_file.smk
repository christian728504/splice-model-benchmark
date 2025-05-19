rule download_file:
    output:
        downloaded_bam = temp(os.path.join(config["OUTPUT_DIR"], "{biosample}", "{file}.bam")),
        downloaded_bam_index = temp(os.path.join(config["OUTPUT_DIR"], "{biosample}", "{file}.bam.bai")),
        filtered_bam = os.path.join(config["OUTPUT_DIR"], "{biosample}", "{file}.filtered.bam"),
        filtered_bam_index = os.path.join(config["OUTPUT_DIR"], "{biosample}", "{file}.filtered.bam.bai")
    priority: 10
    resources:
        download_processes = 1,
        mem_mb=8000
    params:
        url = lambda wildcards: metadata.filter(pl.col("File accession") == wildcards.file)["File download URL"].item(),
        chroms_file = config["CHROMS_FILE"]
    shell:
        """
        wget -O {output.downloaded_bam} {params.url}
        samtools index {output.downloaded_bam}
        samtools view -h {output.downloaded_bam} $(cat {params.chroms_file}) -o {output.filtered_bam}
        samtools index {output.filtered_bam}
        """