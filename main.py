# %%
# Imports
import time
import numpy as np
import pm4py as pm
from preprocessing.preprocessing import preprocessor
from preprocessing.encoding import encoder
from compliance_analysis.aad_experiment.aad.anomaly import detect_anomalies

# %%
# Preprocess data
preprocessed_data = preprocessor.column_rename(
    data_path = '/workspaces/thesis/data/preprocessed/2017_O.gzip',
    case_id_col = 'case:concept:name', activity_col = 'concept:name', 
    timestamp_col = 'time:timestamp', resource_col = 'org:resource')

# %%
# Encode data #REWORK THIS TO INCLUDE ACTUAL ENCODING..
start = time.time()

static_data = encoder.static_encoder(static_cols = preprocessed_data.static_cols)
# dynamic_data = encoder.dynamic_encoder(dynamic_cols = preprocessed_data.dynamic_cols)

end = time.time()
print("Encoding took", end - start, "seconds.")

# %%
# Detect anomalies 
aad = detect_anomalies(static_data, preprocessed_data.data)

# %%
