def run_spliceai():
    from models.spliceai import SpliceAIEvaluator
    evaluator = SpliceAIEvaluator()
    evaluator.get_ground_truth()
    evaluator.generate_spliceai_predictions()
    evaluator.calculate_and_plot_metrics()

def run_segmentnt():
    from models.segmentnt import SegmentNTEvaluator
    evaluator = SegmentNTEvaluator(
        consensus_fasta="reference_files/GCA_000001405.15_GRCh38_no_alt_analysis_set.fasta",
        aurpc_plot_path="results/hg38-baseline-benchmark/segmentnt.svg",
        predicitons_path="results/hg38-baseline-benchmark/segmentnt_predictions.zarr",
    )
    evaluator.get_ground_truth()
    evaluator.generate_segmentnt_predictions()
    evaluator.calculate_and_plot_metrics()

def run_splicetransformer():
    from models.sptransform import SpliceTransformerEvaluator
    evaluator = SpliceTransformerEvaluator(
        consensus_fasta="reference_files/GCA_000001405.15_GRCh38_no_alt_analysis_set.fasta",
        aurpc_plot_path="results/hg38-baseline-benchmark/splicetransformer.svg",
        predicitons_path="results/hg38-baseline-benchmark/splicetransformer_predictions.zarr",
    )
    evaluator.get_ground_truth()
    evaluator.generate_sptransformer_predictions()
    evaluator.calculate_and_plot_metrics()
    
def run_pangolin():
    from models.pangolin import PangolinEvaluator
    evaluator = PangolinEvaluator(
        consensus_fasta="reference_files/GCA_000001405.15_GRCh38_no_alt_analysis_set.fasta",
        aurpc_plot_path="results/hg38-baseline-benchmark/pangolin.svg",
        predicitons_path="results/hg38-baseline-benchmark/pangolin_predictions.zarr",
    )
    evaluator.get_ground_truth()
    evaluator.generate_pangolin_predictions()
    evaluator.calculate_and_plot_metrics()

def main():
    """Run evaluation methods for models (only run one at a time)"""
    run_spliceai()
    
    print("Evaluation completed!")

if __name__ == "__main__":
    main()
