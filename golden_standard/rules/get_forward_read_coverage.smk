rule get_forward_read_coverage:
    input:
        bam_file = os.path.join(config["OUTPUT_DIR"], "{biosample}", "{longread}.filtered.bam")
    output:
        forward_coverage_bed_file = os.path.join(config["OUTPUT_DIR"], "{biosample}", "{longread}.forward.coverage.bg")
    priority: 8
    threads: 32
    resources:
        mem_mb=8000
    script:
        "scripts/get_forward_read_coverage.py"