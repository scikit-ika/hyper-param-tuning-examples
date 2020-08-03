"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
"""

from data_structures.attribute_scheme import AttributeScheme
from classifier.__init__ import *
from drift_detection.__init__ import *
from filters.project_creator import Project
from streams.readers.arff_reader import ARFFReader
from tasks.__init__ import *


# 1. Creating a project
project = Project("projects/single", "sine1")

# 2. Loading an arff file
labels, attributes, stream_records = ARFFReader.read("data_streams/sine1_w_50_n_0.1/sine1_w_50_n_0.1_101.arff")
attributes_scheme = AttributeScheme.get_scheme(attributes)


