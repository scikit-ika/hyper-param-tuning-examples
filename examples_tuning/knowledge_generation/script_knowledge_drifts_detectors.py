# -*- coding: utf-8 -*-

# Script to compute knowledge for the drift detection tuning. 
# Several configurations of the detectors are evaluated on several streams.
# It generates csv files that can be fed to the Pareto analysis to build the knowledge. 


import sys
import os
# Insert here the path to the scikit_multiflow dev version (includes RAM_h measure):
dirpath = os.getcwd()
sys.path.insert(0, dirpath+'\\tornadomaster\\')  # To be modified if on linux : scikit_multiflow_clone_linux


from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.drift_detection.ddm import DDM
from skmultiflow.drift_detection.page_hinkley import PageHinkley

from time import time
from joblib import Parallel, delayed


from tornadomaster.drift_detection.seq_drift2 import SeqDrift2ChangeDetector

from skika.data.bernoulliStream import BernoulliStream
from skika.hyper_parameter_tuning.drift_detectors.evaluate_drift_detection_knowledge import evaluateDriftDetection

# Function to launch in parallel
def Run_trial(stream,listDetectors,namesDetectors,runs):
    print(stream[1])
    eval = evaluateDriftDetection(list_drifts_detectors =listDetectors, list_names_drifts_detectors=namesDetectors, stream = stream[0], n_runs = runs, name_file=stream[1])
    eval.run()

# Varibales 
    
listDetect = [[PageHinkley(min_instances=15, delta=0.005, threshold=1.5, alpha=0.999), PageHinkley(min_instances=15, delta=0.005, threshold=2.5, alpha=0.999)],
                [PageHinkley(min_instances=15, delta=0.005, threshold=1.5, alpha=0.9), PageHinkley(min_instances=15, delta=0.005, threshold=2.5, alpha=0.9)],
                [PageHinkley(min_instances=15, delta=0.005, threshold=0.5, alpha=0.999), PageHinkley(min_instances=15, delta=0.005, threshold=1.5, alpha=0.999)],
                [PageHinkley(min_instances=15, delta=0.005, threshold=0.5, alpha=0.9), PageHinkley(min_instances=15, delta=0.005, threshold=1.5, alpha=0.9)],
                [PageHinkley(min_instances=15, delta=0.05, threshold=1.5, alpha=0.999), PageHinkley(min_instances=15, delta=0.05, threshold=2.5, alpha=0.999)],
                [PageHinkley(min_instances=15, delta=0.05, threshold=1.5, alpha=0.9), PageHinkley(min_instances=15, delta=0.05, threshold=2.5, alpha=0.9)],
                [PageHinkley(min_instances=15, delta=0.05, threshold=0.5, alpha=0.999), PageHinkley(min_instances=15, delta=0.05, threshold=1.5, alpha=0.999)],
                [PageHinkley(min_instances=15, delta=0.05, threshold=0.5, alpha=0.9), PageHinkley(min_instances=15, delta=0.05, threshold=1.5, alpha=0.9)],
                [PageHinkley(min_instances=30, delta=0.005, threshold=1.5, alpha=0.999), PageHinkley(min_instances=30, delta=0.005, threshold=2.5, alpha=0.999)],
                [PageHinkley(min_instances=30, delta=0.005, threshold=1.5, alpha=0.9), PageHinkley(min_instances=30, delta=0.005, threshold=2.5, alpha=0.9)],
                [PageHinkley(min_instances=30, delta=0.005, threshold=0.5, alpha=0.999), PageHinkley(min_instances=30, delta=0.005, threshold=1.5, alpha=0.999)],
                [PageHinkley(min_instances=30, delta=0.005, threshold=0.5, alpha=0.9), PageHinkley(min_instances=30, delta=0.005, threshold=1.5, alpha=0.9)],
                [PageHinkley(min_instances=30, delta=0.05, threshold=1.5, alpha=0.999), PageHinkley(min_instances=30, delta=0.05, threshold=2.5, alpha=0.999)],
                [PageHinkley(min_instances=30, delta=0.05, threshold=1.5, alpha=0.9), PageHinkley(min_instances=30, delta=0.05, threshold=2.5, alpha=0.9)],
                [PageHinkley(min_instances=30, delta=0.05, threshold=0.5, alpha=0.999), PageHinkley(min_instances=30, delta=0.05, threshold=1.5, alpha=0.999)],
                [PageHinkley(min_instances=30, delta=0.05, threshold=0.5, alpha=0.9), PageHinkley(min_instances=30, delta=0.05, threshold=1.5, alpha=0.9)],
                [ADWIN(delta=0.5), ADWIN(delta=0.05)],
                [ADWIN(delta=0.4), ADWIN(delta=0.04)],
                [ADWIN(delta=0.3), ADWIN(delta=0.03)],
                [ADWIN(delta=0.2), ADWIN(delta=0.02)],
                [ADWIN(delta=0.1), ADWIN(delta=0.01)],
                [ADWIN(delta=0.05), ADWIN(delta=0.005)],
                [ADWIN(delta=0.02), ADWIN(delta=0.002)],
                [ADWIN(delta=0.01), ADWIN(delta=0.001)],
                [ADWIN(delta=0.001), ADWIN(delta=0.0001)],
                [DDM(min_num_instances=15, warning_level=0.5, out_control_level=1.0)],
                [DDM(min_num_instances=15, warning_level=0.75, out_control_level=1.25)],
                [DDM(min_num_instances=15, warning_level=1, out_control_level=1.5)],
                [DDM(min_num_instances=15, warning_level=1.25, out_control_level=1.75)],
                [DDM(min_num_instances=15, warning_level=1.5, out_control_level=2.0)],
                [DDM(min_num_instances=30, warning_level=0.5, out_control_level=1.0)],
                [DDM(min_num_instances=30, warning_level=0.75, out_control_level=1.25)],
                [DDM(min_num_instances=30, warning_level=1, out_control_level=1.5)],
                [DDM(min_num_instances=30, warning_level=1.25, out_control_level=1.75)],
                [DDM(min_num_instances=30, warning_level=1.5, out_control_level=2.0)],
                [SeqDrift2ChangeDetector(delta=0.5, block_size=100), SeqDrift2ChangeDetector(delta=0.05, block_size=100)],
                [SeqDrift2ChangeDetector(delta=0.4, block_size=100), SeqDrift2ChangeDetector(delta=0.04, block_size=100)],
                [SeqDrift2ChangeDetector(delta=0.3, block_size=100), SeqDrift2ChangeDetector(delta=0.03, block_size=100)],
                [SeqDrift2ChangeDetector(delta=0.2, block_size=100), SeqDrift2ChangeDetector(delta=0.02, block_size=100)],
                [SeqDrift2ChangeDetector(delta=0.1, block_size=100), SeqDrift2ChangeDetector(delta=0.01, block_size=100)],
                [SeqDrift2ChangeDetector(delta=0.05, block_size=100), SeqDrift2ChangeDetector(delta=0.005, block_size=100)],
                [SeqDrift2ChangeDetector(delta=0.02, block_size=100), SeqDrift2ChangeDetector(delta=0.002, block_size=100)],
                [SeqDrift2ChangeDetector(delta=0.01, block_size=100), SeqDrift2ChangeDetector(delta=0.001, block_size=100)],
                [SeqDrift2ChangeDetector(delta=0.001, block_size=100), SeqDrift2ChangeDetector(delta=0.0001, block_size=100)],
                [SeqDrift2ChangeDetector(delta=0.5, block_size=200), SeqDrift2ChangeDetector(delta=0.05, block_size=200)],
                [SeqDrift2ChangeDetector(delta=0.4, block_size=200), SeqDrift2ChangeDetector(delta=0.04, block_size=200)],
                [SeqDrift2ChangeDetector(delta=0.3, block_size=200), SeqDrift2ChangeDetector(delta=0.03, block_size=200)],
                [SeqDrift2ChangeDetector(delta=0.2, block_size=200), SeqDrift2ChangeDetector(delta=0.02, block_size=200)],
                [SeqDrift2ChangeDetector(delta=0.1, block_size=200), SeqDrift2ChangeDetector(delta=0.01, block_size=200)],
                [SeqDrift2ChangeDetector(delta=0.05, block_size=200), SeqDrift2ChangeDetector(delta=0.005, block_size=200)],
                [SeqDrift2ChangeDetector(delta=0.02, block_size=200), SeqDrift2ChangeDetector(delta=0.002, block_size=200)],
                [SeqDrift2ChangeDetector(delta=0.01, block_size=200), SeqDrift2ChangeDetector(delta=0.001, block_size=200)],
                [SeqDrift2ChangeDetector(delta=0.001, block_size=200), SeqDrift2ChangeDetector(delta=0.0001, block_size=200)]]



namesDetect = ['PH1','PH2','PH3','PH4','PH5','PH6','PH7','PH8','PH9','PH10','PH11','PH12','PH13','PH14','PH15','PH16',
               'ADWIN1','ADWIN2','ADWIN3','ADWIN4','ADWIN5','ADWIN6','ADWIN7','ADWIN8','ADWIN9',
               'DDM1','DDM2','DDM3','DDM4','DDM5','DDM6','DDM7','DDM8','DDM9','DDM10',
               'SeqDrift21','SeqDrift22','SeqDrift23','SeqDrift24','SeqDrift25','SeqDrift26','SeqDrift27','SeqDrift28','SeqDrift29','SeqDrift210',
               'SeqDrift211','SeqDrift212','SeqDrift213','SeqDrift214','SeqDrift215','SeqDrift216','SeqDrift217','SeqDrift218']
# Knowledge
listStreams = [[BernoulliStream(drift_period = 100, n_drifts = 10, widths_drifts = [1], mean_errors = [0.1,0.9]),'BernouW1ME0109'],
                [BernoulliStream(drift_period = 1000, n_drifts = 100, widths_drifts = [1], mean_errors = [0.1,0.8]),'BernouW1ME0108'],
                [BernoulliStream(drift_period = 1000, n_drifts = 100, widths_drifts = [1], mean_errors = [0.2,0.8]),'BernouW1ME0208'],
                [BernoulliStream(drift_period = 1000, n_drifts = 100, widths_drifts = [1], mean_errors = [0.2,0.7]),'BernouW1ME0207'],
                [BernoulliStream(drift_period = 1000, n_drifts = 100, widths_drifts = [1], mean_errors = [0.3,0.7]),'BernouW1ME0307'],
                [BernoulliStream(drift_period = 1000, n_drifts = 100, widths_drifts = [1], mean_errors = [0.3,0.6]),'BernouW1ME0306'],
                [BernoulliStream(drift_period = 1000, n_drifts = 100, widths_drifts = [1], mean_errors = [0.4,0.6]),'BernouW1ME0406'],
                [BernoulliStream(drift_period = 1000, n_drifts = 100, widths_drifts = [1], mean_errors = [0.4,0.5]),'BernouW1ME0405'],
                [BernoulliStream(drift_period = 1000, n_drifts = 100, widths_drifts = [100], mean_errors = [0.1,0.9]),'BernouW100ME0109'],
                [BernoulliStream(drift_period = 1000, n_drifts = 100, widths_drifts = [100], mean_errors = [0.1,0.8]),'BernouW100ME0108'],
                [BernoulliStream(drift_period = 1000, n_drifts = 100, widths_drifts = [100], mean_errors = [0.2,0.8]),'BernouW100ME0208'],
                [BernoulliStream(drift_period = 1000, n_drifts = 100, widths_drifts = [100], mean_errors = [0.2,0.7]),'BernouW100ME0207'],
                [BernoulliStream(drift_period = 1000, n_drifts = 100, widths_drifts = [100], mean_errors = [0.3,0.7]),'BernouW100ME0307'],
                [BernoulliStream(drift_period = 1000, n_drifts = 100, widths_drifts = [100], mean_errors = [0.3,0.6]),'BernouW100ME0306'],
                [BernoulliStream(drift_period = 1000, n_drifts = 100, widths_drifts = [100], mean_errors = [0.4,0.6]),'BernouW100ME0406'],
                [BernoulliStream(drift_period = 1000, n_drifts = 100, widths_drifts = [100], mean_errors = [0.4,0.5]),'BernouW100ME0405'],
                [BernoulliStream(drift_period = 1000, n_drifts = 100, widths_drifts = [500], mean_errors = [0.1,0.9]),'BernouW500ME0109'],
                [BernoulliStream(drift_period = 1000, n_drifts = 100, widths_drifts = [500], mean_errors = [0.1,0.8]),'BernouW500ME0108'],
                [BernoulliStream(drift_period = 1000, n_drifts = 100, widths_drifts = [500], mean_errors = [0.2,0.8]),'BernouW500ME0208'],
                [BernoulliStream(drift_period = 1000, n_drifts = 100, widths_drifts = [500], mean_errors = [0.2,0.7]),'BernouW500ME0207'],
                [BernoulliStream(drift_period = 1000, n_drifts = 100, widths_drifts = [500], mean_errors = [0.3,0.7]),'BernouW500ME0307'],
                [BernoulliStream(drift_period = 1000, n_drifts = 100, widths_drifts = [500], mean_errors = [0.3,0.6]),'BernouW500ME0306'],
                [BernoulliStream(drift_period = 1000, n_drifts = 100, widths_drifts = [500], mean_errors = [0.4,0.6]),'BernouW500ME0406'],
                [BernoulliStream(drift_period = 1000, n_drifts = 100, widths_drifts = [500], mean_errors = [0.4,0.5]),'BernouW500ME0405'],
                [BernoulliStream(drift_period = 1000, n_drifts = 100, widths_drifts = [1], mean_errors = [0.0,1.0]),'BernouW1ME0010'],
                [BernoulliStream(drift_period = 1000, n_drifts = 100, widths_drifts = [1], mean_errors = [0.05,0.95]),'BernouW1ME005095'],
                [BernoulliStream(drift_period = 1000, n_drifts = 100, widths_drifts = [1], mean_errors = [0.05,0.9]),'BernouW1ME00509'],
                [BernoulliStream(drift_period = 1000, n_drifts = 100, widths_drifts = [1], mean_errors = [0.55,0.6]),'BernouW1ME05506'],
                [BernoulliStream(drift_period = 1000, n_drifts = 100, widths_drifts = [100], mean_errors = [0.0,1.0]),'BernouW100ME0010'],
                [BernoulliStream(drift_period = 1000, n_drifts = 100, widths_drifts = [100], mean_errors = [0.05,0.95]),'BernouW100ME005095'],
                [BernoulliStream(drift_period = 1000, n_drifts = 100, widths_drifts = [100], mean_errors = [0.05,0.9]),'BernouW100ME00509'],
                [BernoulliStream(drift_period = 1000, n_drifts = 100, widths_drifts = [100], mean_errors = [0.55,0.6]),'BernouW100ME05506'],
                [BernoulliStream(drift_period = 1000, n_drifts = 100, widths_drifts = [500], mean_errors = [0.0,1.0]),'BernouW500ME0010'],
                [BernoulliStream(drift_period = 1000, n_drifts = 100, widths_drifts = [500], mean_errors = [0.05,0.95]),'BernouW500ME005095'],
                [BernoulliStream(drift_period = 1000, n_drifts = 100, widths_drifts = [500], mean_errors = [0.05,0.9]),'BernouW500ME00509'],
                [BernoulliStream(drift_period = 1000, n_drifts = 100, widths_drifts = [500], mean_errors = [0.55,0.6]),'BernouW500ME05506']]


t1 = time()

Parallel(n_jobs=1, verbose=10)(delayed(Run_trial)(stream = strm, listDetectors=listDetect, namesDetectors=namesDetect, runs=1) for strm in listStreams)
    
t2 = time()
t_tot = t2-t1
t_tot_h = t_tot / 3600

print('Total time: '+str(t_tot_h)+' hours')