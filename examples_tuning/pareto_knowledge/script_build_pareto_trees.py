# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 16:48:09 2020

@author: tlac980
"""

import os

from skika.hyper_parameter_tuning.trees_arf.build_pareto_knowledge_trees import buildTreesKnowledge

"""
 Script to generate pareto fronts from files generated with the knowledge script. 
 
"""

# Names of the evaluated models' names
names = ['ARF10','ARF30','ARF60','ARF70','ARF90','ARF100','ARF120','ARF150','ARF200']    
# Percentages of redundancy in the evaluated streams
perc_redund = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

# Output directory for the output file
output_dir = os.getcwd()
# Path to the file containing the results from the knowledge script
# Results must be processed and shaped like the example files (summary of kappa and RAM per hour performances)
name_file =' /examples/pareto_knowledge/ExamplesTreesKnowledge/Results10-200.csv' # Available in hyper-param-tuning-examples repository

# Processing of the pareto fronts
paretoBuild = buildTreesKnowledge(results_file = name_file, list_perc_redund = perc_redund, list_models = names, output = output_dir, verbose = True)
paretoBuild.load_drift_data()
paretoBuild.calculatePareto()
paretoBuild.bestConfig