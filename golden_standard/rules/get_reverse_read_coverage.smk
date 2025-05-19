rule get_reverse_read_coverage:
    input:
        bam_file = os.path.join(config["OUTPUT_DIR"], "{biosample}", "{longread}.filtered.bam")
    output:
        reverse_coverage_bed_file = os.path.join(config["OUTPUT_DIR"], "{biosample}", "{longread}.reverse.coverage.bg")
    priority: 7
    threads: 32
    resources:
        mem_mb=8000
    script:
        "scripts/get_reverse_read_coverage.py"