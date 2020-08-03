# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:07:45 2020

@author: tlac980
"""

import os
import csv
from skika.hyper_parameter_tuning.drift_detectors.build_pareto_knowledge_drifts import buildDriftKnowledge

dirpath = os.getcwd()
directoryPathFiles = dirpath+'\\ResultsDriftKnowledge'

namesStm = ['BernouW1ME0010','BernouW1ME005095','BernouW1ME00509','BernouW1ME0109','BernouW1ME0108','BernouW1ME0208','BernouW1ME0207','BernouW1ME0307','BernouW1ME0306','BernouW1ME0406','BernouW1ME0506','BernouW1ME05506',
            'BernouW100ME0010','BernouW100ME005095','BernouW100ME00509','BernouW100ME0109','BernouW100ME0108','BernouW100ME0208','BernouW100ME0207','BernouW100ME0307','BernouW100ME0306','BernouW100ME0406','BernouW100ME0506','BernouW100ME05506',
            'BernouW500ME0010','BernouW500ME005095','BernouW500ME00509','BernouW500ME0109','BernouW500ME0108','BernouW500ME0208','BernouW500ME0207','BernouW500ME0307','BernouW500ME0306','BernouW500ME0406','BernouW500ME0506','BernouW500ME05506']

namesDetect = [['PH1','PH2','PH3','PH4','PH5','PH6','PH7','PH8','PH9','PH10','PH11','PH12','PH13','PH14','PH15','PH16'],
                          ['ADWIN1','ADWIN2','ADWIN3','ADWIN4','ADWIN5','ADWIN6','ADWIN7','ADWIN8','ADWIN9'],
                          ['DDM1','DDM2','DDM3','DDM4','DDM5','DDM6','DDM7','DDM8','DDM9','DDM10'],
                          ['SeqDrift21','SeqDrift22','SeqDrift23','SeqDrift24','SeqDrift25','SeqDrift26','SeqDrift27','SeqDrift28','SeqDrift29','SeqDrift210',
                           'SeqDrift211','SeqDrift212','SeqDrift213','SeqDrift214','SeqDrift215','SeqDrift216','SeqDrift217','SeqDrift218']]

paretoBuild = buildDriftKnowledge(results_directory = directoryPathFiles, namesDetectors = namesDetect, namesStreams = namesStm , verbose =True)
paretoBuild.load_drift_data()
paretoBuild.calculatePareto()
config = paretoBuild.bestConfig

with open('bestConfigsDrifts.csv','w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(config)

paretoBuild.processMetaFeatures()