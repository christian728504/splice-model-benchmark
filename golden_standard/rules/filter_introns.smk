rule filter_introns:
    input:
        bed_file = os.path.join(config["OUTPUT_DIR"], "{biosample}", "{longread}.introns.tab")
    output:
        inclusive_introns_bed = os.path.join(config["OUTPUT_DIR"], "{biosample}", "{longread}.inclusive.introns.filtered.tab"),
        exclusive_introns_bed = os.path.join(config["OUTPUT_DIR"], "{biosample}", "{longread}.exclusive.introns.filtered.tab")
    priority: 5
    params:
        gtf_parquet_path = config["GTF_PARQUET_PATH"]
    script:
        "scripts/filter_introns.py"