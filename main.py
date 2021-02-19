# %%
# Imports
import time

from preprocessing.preprocessing import preprocessing
from pm4py.objects.log.util import get_log_representation
from compliance_analysis.aad_experiment.aad.anomaly import detect_anomalies

# %%
# Preprocess data
preprocessed_data, log = preprocessing.column_rename(
    data_path = '/workspaces/thesis/data/preprocessed/2017_O.gzip',
    log_path = '/workspaces/thesis/data/raw/BPI Challenge 2017.xes.gz',
    case_id_col = 'case:concept:name', activity_col = 'concept:name', 
    timestamp_col = 'time:timestamp', resource_col = 'org:resource')

# %%
# Encode data
start = time.time()
encoded_data, feature_names = get_log_representation.get_representation(
    log, 
    str_ev_attr=["concept:name", "org:resource"],
    str_tr_attr=[],#list(preprocessed_data.str_cols.columns), 
    num_ev_attr=[], 
    num_tr_attr=[],#list(preprocessed_data.static_num_cols.columns), 
    str_evsucc_attr=[])#["concept:name", "org:resource"])
end = time.time()
print("Encoding took", end - start, "seconds.")

# %%
# Detect anomalies 
model, x_transformed, queried, ridxs_counts, region_extents = detect_anomalies(encoded_data[:100,:], log, preprocessed_data.data)

# %%