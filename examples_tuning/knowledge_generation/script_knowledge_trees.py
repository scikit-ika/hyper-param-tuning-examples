# Imports
import os

from skmultiflow.meta import AdaptiveRandomForest
from skmultiflow.drift_detection import ADWIN
from joblib import Parallel, delayed

from skika.hyper_parameter_tuning.trees_arf.evaluate_prequential_and_adapt import EvaluatePrequentialAndAdaptTreesARF
from skika.data.random_rbf_generator_redund import RandomRBFGeneratorRedund


####################################
"""
 Script to generate meta_knowledge files for tuning the number of trees in ARF. 
 This script evaluates multiple ARF with different number of trees on several streams.
 It outputs csv files that can be then processed to get knowledge from Pareto front.
"""
# TODO : Automatise getting the results instead of processing cvs files

# FUNCTION TO EVELUATE A STREAM WITH A LIST OF MODELS
def EvaluateModels(models_list, models_names, data_stream) :
    data_stream[0].prepare_for_use()

    try :
        dirpath = os.getcwd()
        new_dir_path = dirpath+'\\KnowledgeRandomRBFTrees'
        os.mkdir(new_dir_path)
        file_name = new_dir_path+'\\'+data_stream[1]+'.csv'
    except FileExistsError :
        file_name = new_dir_path+'\\'+data_stream[1]+'.csv'

    # Setup the evaluator
    evaluator = EvaluatePrequentialAndAdaptTreesARF(metrics=['accuracy','kappa','running_time','ram_hours'],
                                    show_plot=False,
                                    pretrain_size=200,
                                    max_samples=2000,
                                    output_file = file_name)

    # Run evaluation
    model = evaluator.evaluate(stream=data_stream[0], model=models_list, model_names=models_names)
    return model


# List of streams
streams = [[RandomRBFGeneratorRedund(model_random_state=None, sample_random_state=None, n_classes=4, n_features=30, perc_redund_feature = 0.1, n_centroids=100),'StreamRBFmodif10p'],
     [RandomRBFGeneratorRedund(model_random_state=None, sample_random_state=None, n_classes=4, n_features=30, perc_redund_feature = 0.2, n_centroids=100),'StreamRBFmodif20p'],
     [RandomRBFGeneratorRedund(model_random_state=None, sample_random_state=None, n_classes=4, n_features=30, perc_redund_feature = 0.3, n_centroids=100),'StreamRBFmodif30p'],
     [RandomRBFGeneratorRedund(model_random_state=None, sample_random_state=None, n_classes=4, n_features=30, perc_redund_feature = 0.4, n_centroids=100),'StreamRBFmodif40p'],
     [RandomRBFGeneratorRedund(model_random_state=None, sample_random_state=None, n_classes=4, n_features=30, perc_redund_feature = 0.5, n_centroids=100),'StreamRBFmodif50p'],
     [RandomRBFGeneratorRedund(model_random_state=None, sample_random_state=None, n_classes=4, n_features=30, perc_redund_feature = 0.6, n_centroids=100),'StreamRBFmodif60p'],
     [RandomRBFGeneratorRedund(model_random_state=None, sample_random_state=None, n_classes=4, n_features=30, perc_redund_feature = 0.7, n_centroids=100),'StreamRBFmodif70p'],
     [RandomRBFGeneratorRedund(model_random_state=None, sample_random_state=None, n_classes=4, n_features=30, perc_redund_feature = 0.8, n_centroids=100),'StreamRBFmodif80p'],
     [RandomRBFGeneratorRedund(model_random_state=None, sample_random_state=None, n_classes=4, n_features=30, perc_redund_feature = 0.9, n_centroids=100),'StreamRBFmodif90p']]

# Models to evaluate
arf1 = AdaptiveRandomForest(n_estimators = 10, lambda_value=6, grace_period=10, split_confidence=0.1, tie_threshold=0.005, warning_detection_method= ADWIN(delta=0.01), drift_detection_method=ADWIN(delta=0.001))
arf2 = AdaptiveRandomForest(n_estimators = 70, lambda_value=6, grace_period=10, split_confidence=0.1, tie_threshold=0.005, warning_detection_method= ADWIN(delta=0.01), drift_detection_method=ADWIN(delta=0.001))
arf3 = AdaptiveRandomForest(n_estimators = 80, lambda_value=6, grace_period=10, split_confidence=0.1, tie_threshold=0.005, warning_detection_method= ADWIN(delta=0.01), drift_detection_method=ADWIN(delta=0.001))
arf4 = AdaptiveRandomForest(n_estimators = 90, lambda_value=6, grace_period=10, split_confidence=0.1, tie_threshold=0.005, warning_detection_method= ADWIN(delta=0.01), drift_detection_method=ADWIN(delta=0.001))
arf5 = AdaptiveRandomForest(n_estimators = 100, lambda_value=6, grace_period=10, split_confidence=0.1, tie_threshold=0.005, warning_detection_method= ADWIN(delta=0.01), drift_detection_method=ADWIN(delta=0.001))
arf6 = AdaptiveRandomForest(n_estimators = 110, lambda_value=6, grace_period=10, split_confidence=0.1, tie_threshold=0.005, warning_detection_method= ADWIN(delta=0.01), drift_detection_method=ADWIN(delta=0.001))
arf7 = AdaptiveRandomForest(n_estimators = 120, lambda_value=6, grace_period=10, split_confidence=0.1, tie_threshold=0.005, warning_detection_method= ADWIN(delta=0.01), drift_detection_method=ADWIN(delta=0.001))

#modelsList = [arf1]
modelsList = [arf1,arf2,arf3,arf4,arf5,arf6,arf7]
#modelsNames = ['ARF1']
modelsNames = ['ARF1','ARF2','ARF3','ARF4','ARF5','ARF6','ARF7']

Parallel(n_jobs=3, verbose=10)(delayed(EvaluateModels)(modelsList, modelsNames, stream) for stream in streams)
