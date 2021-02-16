# %%
<<<<<<< HEAD
# Adding changes to commit: git commit -m "name or push"
# Pushing changes to github: git push -u origin main
# %%
# Imports
import time
import numpy as np
import pandas as pd
import pm4py as pm
import pyarrow as pa
=======
# Imports
import time
>>>>>>> 43304ca41a6ba35a03ba2f00f2a6fd5bc5b61b4c
from preprocessing.preprocessing import preprocessing
from pm4py.objects.log.util import get_log_representation
from compliance_analysis.aad_experiment.aad.anomaly import detect_anomalies

# %%
# Preprocess data
preprocessed_data, log = preprocessing.column_rename(
    '/workspaces/thesis/data/preprocessed/2012_A.gzip', 
    case_id_col = 'case:concept:name', activity_col = 'concept:name', 
    timestamp_col = 'time:timestamp', resource_col = 'org:resource')

# %%
# Encode data
start = time.time()
encoded_data, feature_names = get_log_representation.get_representation(
    log, 
    str_ev_attr=["concept:name", "org:resource"],
    str_tr_attr=list(preprocessed_data.str_cols.drop(columns=['concept:name','org:resource']).columns), 
    num_ev_attr=list(preprocessed_data.num_cols.drop(columns=['Unnamed: 0']).columns), 
    num_tr_attr=[], 
    str_evsucc_attr=["concept:name", "org:resource"])
end = time.time()
print("Encoding took", end - start, "seconds.")

# %%
# Detect anomalies 
<<<<<<< HEAD
model, x_transformed, queried, ridxs_counts, region_extents = detect_anomalies(encoded_data[:10,:], log)
# %%
model
# %%
=======
model, x_transformed, queried, ridxs_counts, region_extents = detect_anomalies(encoded_data[:10000,:], log)
# %%
>>>>>>> 43304ca41a6ba35a03ba2f00f2a6fd5bc5b61b4c
