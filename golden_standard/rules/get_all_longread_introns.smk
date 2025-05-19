rule get_all_longread_introns:
    input:
        bam_files = expand(
            os.path.join(config["OUTPUT_DIR"], "{biosample}", "{longread}.filtered.bam"),
            zip,
            biosample=long_read_biosamples,
            longread=long_read_accessions
        )
    output:
        introns = os.path.join("all_longread_introns.tab")
    priority: 6
    threads: 28
    params:
        max_mismatch_rate = 0.05,
        min_mapq = 30,
        metadata_file = config["METADATA_FILE"],
    script:
        "scripts/get_alls_introns.py"