rule get_introns_bed:
    input:
        bam_file = os.path.join(config["OUTPUT_DIR"], "{biosample}", "{longread}.filtered.bam")
    output:
        bed_file = os.path.join(config["OUTPUT_DIR"], "{biosample}", "{longread}.introns.tab")
    priority: 6
    threads: 8
    params:
        max_mismatch_rate = 0.05,
        min_mapq = 30,
    script:
        "scripts/get_introns.py"