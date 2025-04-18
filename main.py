# from models.spliceai import SpliceAIEvaluator

# def main():
#     evaluator = SpliceAIEvaluator(
#         aurpc_plot_path="results/spliceai_gtf_v29_nofilter.png",
#         gencode_gtf="reference_files/gencode.v29.basic.annotation.gtf.parquet",
#         filter_transcripts=False
#     )
#     evaluator.get_ground_truth()
#     evaluator.generate_spliceai_predictions()
#     evaluator.calculate_and_plot_metrics()

from models.segmentnt import SegmentNTEvaluator

def main():
    evaluator = SegmentNTEvaluator(
        aurpc_plot_path="results/segmentnt_siteonly.png",
        consensus_fasta="reference_files/GCA_000001405.15_GRCh38_no_alt_analysis_set.fasta"
    )
    evaluator.get_ground_truth()
    evaluator.generate_segmentnt_predictions()
    evaluator.calculate_and_plot_metrics()

if __name__ == "__main__":
    main()
