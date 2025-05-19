rule get_all_shortread_introns:
    input:
        bam_files = expand(
            os.path.join(config["OUTPUT_DIR"], "{biosample}", "{shortread}.filtered.bam"),
            zip,
            biosample=short_read_biosamples,
            shortread=short_read_accessions
        )
    output:
        introns = os.path.join("all_shortread_introns.tab")
    priority: 6
    threads: 28
    params:
        max_mismatch_rate = 0.05,
        min_mapq = 30,
        metadata_file = config["METADATA_FILE"],
    script:
        "scripts/get_all_introns.py"