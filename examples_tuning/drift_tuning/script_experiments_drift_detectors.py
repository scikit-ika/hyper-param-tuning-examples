# -*- coding: utf-8 -*-

# Script to experiment the adaptive drift tuning against non adaptive settings

import itertools
import sys
import os
# Insert here the path to the tornado framework:
dirpath = os.getcwd()
sys.path.insert(0, dirpath+'\\tornadomaster\\')

import csv
#from itertools import zip_longest

from time import time

from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.drift_detection.ddm import DDM
from skmultiflow.drift_detection.page_hinkley import PageHinkley

from tornadomaster.drift_detection.seq_drift2 import SeqDrift2ChangeDetector

from skika.data.bernoulli_stream import BernoulliStream
from skika.hyper_parameter_tuning.drift_detectors.evaluate_drift_detection_experiment import evaluateDriftDetection

from joblib import Parallel, delayed

########################################################################################


def RunDriftExperiment(drift_detect_eval, names_drift_detect_eval, names_files, metaK_bases, win_size, data_stream):
    
    for drift_detec, name_drift_detec, name, metaK in zip(drift_detect_eval, names_drift_detect_eval, names_files, metaK_bases) :
        eval = evaluateDriftDetection(list_drifts_detectors = drift_detec,
                         list_names_drifts_detectors =  name_drift_detec,
                         adapt = [0, 0, 1],
                         k_base = metaK,
                         dict_k_base = dictionary,
                         win_adapt_size = win_size,
                         stream = data_stream[0],
                         n_runs = 1,
                         name_file = data_stream[1]+'_'+name+'_WinSize'+str(win_size))

        eval.run()

#######################################################################################################################################
#### Construction of the meta-Knowledge base ####
        
# Meta-features in the knowledge base
sev = [1,100,500]
mag = [1.0,0.751,0.725,0.632,0.578,0.447,0.389,0.289,0.227,0.142,0.067,0.033]

meta_features = list(itertools.product(sev,mag))

# Load best configurations from Pareto
with open(dirpath+'/bestConfigs.csv') as csvDataFile:
    best_configs = [row for row in csv.reader(csvDataFile)]

# Build dict to link configs names and drift detectors
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

dictionary = dict(zip(namesDetect, listDetect))
#######################################################################################################################################

# Detectors configurations for evaluation
drift_detect_eval = [[[PageHinkley(min_instances=30, delta=0.005, threshold=1.5, alpha=0.999), PageHinkley(min_instances=30, delta=0.005, threshold=0.5, alpha=0.999)],
                        [PageHinkley(min_instances=30, delta=0.005, threshold=2.5, alpha=0.9), PageHinkley(min_instances=30, delta=0.005, threshold=1.5, alpha=0.9)],
                        [PageHinkley(min_instances=30, delta=0.005, threshold=2.5, alpha=0.9), PageHinkley(min_instances=30, delta=0.005, threshold=1.5, alpha=0.9)]],
                    [[ADWIN(delta=0.2), ADWIN(delta=0.02)],
                        [ADWIN(delta=0.5), ADWIN(delta=0.05)],
                        [ADWIN(delta=0.5), ADWIN(delta=0.05)]],
                    [[DDM(min_num_instances=30, warning_level=2.0, out_control_level=3.0)],
                        [DDM(min_num_instances=15, warning_level=1.25, out_control_level=1.75)],
                        [DDM(min_num_instances=15, warning_level=1.25, out_control_level=1.75)]],
                    [[SeqDrift2ChangeDetector(delta=0.1, block_size=200), SeqDrift2ChangeDetector(delta=0.01, block_size=200)],
                        [SeqDrift2ChangeDetector(delta=0.001, block_size=200), SeqDrift2ChangeDetector(delta=0.0001, block_size=200)],
                        [SeqDrift2ChangeDetector(delta=0.001, block_size=200), SeqDrift2ChangeDetector(delta=0.0001, block_size=200)]]]

names_drift_detect_eval = [['PH9','PH10','PH10Adapt'],
                           ['ADWIN4','ADWIN1','ADWIN1'],
                           ['DDMRef','DDM4','DDM4'],
                           ['SeqDrift214','SeqDrift218','SeqDrift218']]


# Build metaK bases : Select best configurations depending on what detector is used
metaK_bases = []
list_ind_detec = [0,1,2,3] # TODO : MODIFY depending what detector is used -> 0 : PH, 1 : ADWIN, 2: DDM, 3: SeqDrift2
for ind_detec in list_ind_detec :
    metaK_bases.append([meta_features,list(map(list, zip(*best_configs)))[ind_detec]])

names_files = ['testAdaptEvalPH','testAdaptEvalADWIN','testAdaptEvalDDM','testAdaptEvalSeqDrift2']

win_sizes= [2,4,6,8,10,20,30,40,50]

streams = [[BernoulliStream(drift_period=1500, n_drifts = 300, widths_drifts = [1,100,500], mean_errors = [[0.1,0.9],[0.5,0.6],[0.2,0.8],[0.4,0.5],[0.3,0.7]], n_stable_drifts = 2),"Bernou2"],
           [BernoulliStream(drift_period=1500, n_drifts = 300, widths_drifts = [1,100,500], mean_errors = [[0.1,0.9],[0.5,0.6],[0.2,0.8],[0.4,0.5],[0.3,0.7]], n_stable_drifts = 4),"Bernou4"],
           [BernoulliStream(drift_period=1500, n_drifts = 300, widths_drifts = [1,100,500], mean_errors = [[0.1,0.9],[0.5,0.6],[0.2,0.8],[0.4,0.5],[0.3,0.7]], n_stable_drifts = 6),"Bernou6"],
           [BernoulliStream(drift_period=1500, n_drifts = 300, widths_drifts = [1,100,500], mean_errors = [[0.1,0.9],[0.5,0.6],[0.2,0.8],[0.4,0.5],[0.3,0.7]], n_stable_drifts = 8),"Bernou8"],
           [BernoulliStream(drift_period=1500, n_drifts = 300, widths_drifts = [1,100,500], mean_errors = [[0.1,0.9],[0.5,0.6],[0.2,0.8],[0.4,0.5],[0.3,0.7]], n_stable_drifts = 10),"Bernou10"],
           [BernoulliStream(drift_period=1500, n_drifts = 300, widths_drifts = [1,100,500], mean_errors = [[0.1,0.9],[0.5,0.6],[0.2,0.8],[0.4,0.5],[0.3,0.7]], n_stable_drifts = 20),"Bernou20"],
           [BernoulliStream(drift_period=1500, n_drifts = 300, widths_drifts = [1,100,500], mean_errors = [[0.1,0.9],[0.5,0.6],[0.2,0.8],[0.4,0.5],[0.3,0.7]], n_stable_drifts = 30),"Bernou30"],
           [BernoulliStream(drift_period=1500, n_drifts = 300, widths_drifts = [1,100,500], mean_errors = [[0.1,0.9],[0.5,0.6],[0.2,0.8],[0.4,0.5],[0.3,0.7]], n_stable_drifts = 40),"Bernou40"],
           [BernoulliStream(drift_period=1500, n_drifts = 300, widths_drifts = [1,100,500], mean_errors = [[0.1,0.9],[0.5,0.6],[0.2,0.8],[0.4,0.5],[0.3,0.7]], n_stable_drifts = 50),"Bernou50"]]

combin = list(itertools.product(win_sizes, streams))

t1 = time()


Parallel(n_jobs=1, verbose=10)(delayed(RunDriftExperiment)(drift_detect_eval=drift_detect_eval, names_drift_detect_eval=names_drift_detect_eval, names_files=names_files, metaK_bases=metaK_bases, win_size=c[0], data_stream = c[1]) for c in combin)


t2 = time()

print('Done in : {}'.format(t2-t1))