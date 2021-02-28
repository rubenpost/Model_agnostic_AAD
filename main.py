# %%
# Imports
import time
import importlib
import pandas as pd
import numpy as np
import pm4py as pm
from aad_experiment.common.utils import dataframe_to_matrix
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

static_num_cols = pd.concat([preprocessed_data.static_num_cols, preprocessed_data.case_id_col], axis=1)
encoded_snc = static_num_cols.groupby(['case:concept:name'], as_index=False).agg(['max'])
dummies = pd.get_dummies(preprocessed_data.static_cat_cols[['EventOrigin', 'lifecycle:transition', 'LoanGoal', 'ApplicationType']])
static_cat = pd.concat([preprocessed_data.case_id_col, dummies], axis=1)
encoded_scc = static_cat.groupby(['case:concept:name'], as_index=False).agg(['max'])
static_data = pd.concat([encoded_snc, encoded_scc], axis=1)
dummies = pd.get_dummies(preprocessed_data.dynamic_cat_cols[['Action', 'org:resource', 'concept:name']])
dynamic_cat = pd.concat([preprocessed_data.case_id_col, dummies], axis=1)
encoded_dcc = dynamic_cat.groupby(['case:concept:name'], as_index=False).agg(['max'])
dynamic_data = pd.concat([encoded_dcc, encoded_dcc], axis=1)

encoded_data = pd.concat([static_data, encoded_dcc], axis=1)
encoded_data = np.asarray(encoded_data)

end = time.time()
print("Encoding took", end - start, "seconds.")

# %%
# Detect anomalies 
aad = detect_anomalies(np.asarray(data), preprocessed_data)

# %%
scored = aad[1].decision_function(data)

# %%
# %%
aad[0]
# %%
a = pd.DataFrame(scored)
a.sort_values(by=0,ascending=False)
# %%
a
# %%
from sklearn.ensemble import IsolationForest
model=IsolationForest()
model.fit(dataframe_to_matrix)
b = pd.DataFrame()
b["scores"] = model.decision_function(data)
# %%
print(b.sort_values(by='scores', ascending=False))
print(a.sort_values(by=0,ascending=False))
# %%
pd.set_option("display.max_rows", 999)
# %%
a['index'] = a.index
a['index b'] = b.index
# %%
test = a['index'] == a['index b']
# %%
test.value_counts()
# %%
a['scores b'] = b['scores']
# %%
np.append(encoded_data, np.asarray(a))
# %%
a[0]
# %%
encoded_data
# %%
encoded_data.shape
# %%
a
# %%
data = pd.DataFrame(encoded_data)
# %%
a
# %%
data['score'] = a
# %%
data
# %%
aad[1].get_params()
# %%
