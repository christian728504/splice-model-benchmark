rule get_bam_stats:
    input:
        bam_file = os.path.join(config["OUTPUT_DIR"], "{biosample}", "{longread}.filtered.bam")
    output:
        stats_file = os.path.join(config["OUTPUT_DIR"], "{biosample}", "{longread}.bam.stats"),
        plot = os.path.join(config["OUTPUT_DIR"], "{biosample}", "{longread}.svg"),
    priority: 9
    threads: 8
    script:
        "scripts/get_bam_stats.py"