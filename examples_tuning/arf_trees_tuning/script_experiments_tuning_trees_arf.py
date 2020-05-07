import os

from skmultiflow.meta import AdaptiveRandomForest
from skmultiflow.drift_detection import ADWIN

from skika.data.random_rbf_generator_redund import RandomRBFGeneratorRedund
from skika.data.hyper_plane_generator_redund import HyperplaneGeneratorRedund
from skika.data.stream_generator_redundancy_drift import StreamGeneratorRedund
from skika.hyper_parameter_tuningevaluate_prequential_and_adapt import EvaluatePrequentialAndAdaptTreesARF


from joblib import Parallel, delayed

#############################
# Function for parallelisation
def EvaluateModels(stream, run, n_trees, n_samples_max, n_samples_meas, metaModel = None): # For adaptive experiments

    stream[0].prepare_for_use()


    # Evaluate model (with adaptation or not)
    arf = AdaptiveRandomForest(n_estimators = n_trees, lambda_value=6, grace_period=10, split_confidence=0.1, tie_threshold=0.005, warning_detection_method= ADWIN(delta=0.01), drift_detection_method=ADWIN(delta=0.001))

    modelsList = [arf]
    modelsNames = ['ARF']
    
    try :
        dirpath = os.getcwd()
        new_dir_path = dirpath+'/ExperimentTuningTrees'
        os.mkdir(new_dir_path)
    except FileExistsError :
        pass
        
    if metaModel == None :
        fileName = new_dir_path+'\\'+stream[1]+'_'+str(n_trees)+'Trees_Adapt_Drift_ARDClassic_Run'+str(run)+'.csv'
    else :
        fileName = new_dir_path+'\\'+stream[1]+'_AdativeSetting_Trees_Adapt_Drift_ARDClassic_Run'+str(run)+'.csv'
    

    evaluator = EvaluatePrequentialAndAdaptTreesARF(metrics=['accuracy','kappa','running_time','ram_hours'],
                                    show_plot=False,
                                    n_wait=n_samples_meas,
                                    pretrain_size=200,
                                    max_samples=n_samples_max,
                                    output_file = fileName,
                                    metaKB=metaModel)

    # Run evaluation
    evaluator.evaluate(stream=stream[0], model=modelsList, model_names=modelsNames)

########################################################
# Load the meta-model
dictMeta = {0.0:60 ,0.1:30, 0.2:30, 0.3:30, 0.4:60, 0.5:70, 0.6:60, 0.7:30, 0.8:30, 0.9:30} # dict = {'pourc redund feat':best nb tree}

## Example 1 : 1 adaptive run with RandomRBFRedund 10k, 10 drifts.
## Number of initial trees = 10
#
n_run = 1
nb_trees_init = 10
stream= [StreamGeneratorRedund(base_stream = RandomRBFGeneratorRedund(n_classes=2, n_features=30, n_centroids=50, noise_percentage = 0.0), random_state=None, n_drifts = 10, n_instances = 10000),'RandomRBF']
EvaluateModels(stream = stream, run = n_run, n_trees = nb_trees_init, n_samples_max = 10000, n_samples_meas = 500, metaModel=dictMeta)
# ###########

## Example 2 : 1 adaptive run with Electricity dataset
## Number of initial trees = 10
#
# nb_runs = 1
# 
# stream = [data.FileStream(./third_party/scikit_multiflow/src/skmultiflow/data/datasets/elec.csv'), 'Electricity']
#
# nb_trees_init = 10
# EvaluateModels(stream = stream, run = n_run, n_trees = nb_trees_init, n_samples_max = 45312, n_samples_meas = 200, metaModel=dictMeta)
# ###########

## Example 3 : 1 adaptive run with CovType dataset
## Number of initial trees = 10
#
# nb_runs = 1
# 
# stream = [data.FileStream(./third_party/scikit_multiflow/src/skmultiflow/data/datasets/covtype.csv'), 'CoverType']
#
# nb_trees_init = 10
# EvaluateModels(stream = stream, run = n_run, n_trees = nb_trees_init, n_samples_max = 45312, n_samples_meas = 200, metaModel=dictMeta)
# ###########

## Example 4 : 30 parallelised runs RandomRBF & Hyperplan 100k, 100 drifts, non adative with 10, 60 and 120 trees 
## + 30 adaptive parallelised runs with number of initial trees = 10
#
# nb_runs = 30
# nbTrees = [10,60,120]
# n_cores = 4
# streams= [[StreamGeneratorRedund(base_stream = RandomRBFGeneratorRedund(n_classes=2, n_features=30, n_centroids=50, noise_percentage = 0.0), random_state=None, n_drifts = 100, n_instances = 100000),'RandomRBF'],
#            [StreamGeneratorRedund(base_stream = HyperplaneGeneratorRedund(n_features = 30, n_drift_features = 0, noise_percentage = 0.0), random_state=None, n_drifts = 100, n_instances = 100000), 'HyperPlan']]
# for datastream in streams :
#     for nbTreesInit in nbTrees :
#         Parallel(n_jobs=n_cores, verbose=10)(delayed(EvaluateModels)(stream = datastream, run = n_run, n_trees = nbTreesInit, n_samples_max = 100000, n_samples_meas = 500, metaModel=None) for n_run in range(nb_runs))
#
#     nbTreesInit = 10
#     Parallel(n_jobs=n_cores, verbose=10)(delayed(EvaluateModels)(stream = datastream, run = n_run, n_trees = nbTreesInit, n_samples_max = 100000, n_samples_meas = 500, metaModel=dictMeta) for n_run in range(nb_runs))
# ###########

## Example 5 : 30 parallelised runs Electricity dataset, non adative with 10, 60 and 120 trees 
## + 30 adaptive parallelised runs with number of initial trees = 10
#
# nb_runs = 30
# nbTrees = [10,60,120]
# n_cores = 4
# datastream = [data.FileStream(./third_party/scikit_multiflow/src/skmultiflow/data/datasets/elec.csv'), 'Covtype']
# for nbTreesInit in nbTrees :
#     Parallel(n_jobs=n_cores, verbose=10)(delayed(EvaluateModels)(stream = datastream, run = n_run, n_trees = nbTreesInit, n_samples_max = 45312, n_samples_meas = 200, metaModel=None) for n_run in range(nb_runs))
#
# nbTreesInit = 10
# Parallel(n_jobs=n_cores, verbose=10)(delayed(EvaluateModels)(stream = datastream, run = n_run, n_trees = nbTreesInit, n_samples_max = 45312, n_samples_meas = 200, metaModel=dictMeta) for n_run in range(nb_runs))
# ###########

