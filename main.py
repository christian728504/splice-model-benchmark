def run_spliceai():
    from models.spliceai import SpliceAIEvaluator
    evaluator = SpliceAIEvaluator()
    evaluator.filter_gencode()
    # evaluator.get_ground_truth()
    # evaluator.generate_spliceai_predictions()
    evaluator.calculate_and_plot_metrics_stratified()

def run_segmentnt():
    from models.segmentnt import SegmentNTEvaluator
    evaluator = SegmentNTEvaluator()
    evaluator.filter_gencode()
    evaluator.get_ground_truth()
    evaluator.generate_segmentnt_predictions()
    evaluator.calculate_and_plot_metrics_stratified()

def run_splicetransformer():
    from models.sptransform import SpliceTransformerEvaluator
    evaluator = SpliceTransformerEvaluator()
    evaluator.filter_gencode()
    evaluator.get_ground_truth()
    evaluator.generate_sptransformer_predictions()
    evaluator.calculate_and_plot_metrics_stratified()
    
def run_pangolin():
    from models.pangolin import PangolinEvaluator
    evaluator = PangolinEvaluator()
    evaluator.filter_gencode()
    evaluator.get_ground_truth()
    evaluator.generate_pangolin_predictions()
    evaluator.calculate_and_plot_metrics_stratified()

def main():
    """Run evaluation methods for models (only run one at a time)"""
    run_spliceai()
    
if __name__ == "__main__":
    main()
