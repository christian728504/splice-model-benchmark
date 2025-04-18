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
    evaluator = SegmentNTEvaluator()
    evaluator.get_ground_truth()
    evaluator.generate_segmentnt_predictions()
    evaluator.calculate_and_plot_metrics()

if __name__ == "__main__":
    main()
