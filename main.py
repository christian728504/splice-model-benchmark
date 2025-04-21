from utils import run_with_logging

def run_spliceai():
    from models.spliceai import SpliceAIEvaluator
    evaluator = SpliceAIEvaluator()
    evaluator.get_ground_truth()
    evaluator.generate_spliceai_predictions()
    evaluator.calculate_and_plot_metrics()

def run_segmentnt():
    from models.segmentnt import SegmentNTEvaluator
    evaluator = SegmentNTEvaluator(
        aurpc_plot_path='results/5000bpwindow_segmentnt.svg'
    )
    evaluator.get_ground_truth()
    evaluator.generate_segmentnt_predictions()
    evaluator.calculate_and_plot_metrics()

def run_splicetransformer():
    from models.sptransform import SpliceTransformerEvaluator
    evaluator = SpliceTransformerEvaluator()
    evaluator.get_ground_truth()
    evaluator.generate_sptransformer_predictions()
    evaluator.calculate_and_plot_metrics()
    
def run_pangolin():
    from models.pangolin import PangolinEvaluator
    evaluator = PangolinEvaluator()
    evaluator.get_ground_truth()
    evaluator.generate_pangolin_predictions()
    evaluator.calculate_and_plot_metrics()

def main():
    """Run evaluation methods for models (only run one at a time)"""
    # run_with_logging(run_spliceai, "spliceai.log")
    run_with_logging(run_segmentnt, "segmentnt.log")
    # run_with_logging(run_splicetransformer, "sptransformer.log")
    # run_with_logging(run_pangolin, "pangolin.log")
    
    print("All evaluations completed!")

if __name__ == "__main__":
    main()
