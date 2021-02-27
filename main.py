# %%
# Imports
import time
import importlib
import pandas as pd
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
aad = detect_anomalies(encoded_data, preprocessed_data)

# %%
# %%
encoded_data.dtypes.unique()
# %%
preprocessed_data.dynamic_num_cols
# %%
np.set_printoptions(suppress=True)
encoded_data
# %%
for column in preprocessed_data.data.columns:
    if preprocessed_data.data[column].nunique() == 2 or preprocessed_data.data[column].nunique() == 1:
        print(column)
# %%
preprocessed_data.str_cols.nunique()#.columns
# %%
df = pd.read_parquet('/workspaces/thesis/data/preprocessed/2017_O.gzip')
# %%
df.dtypes
# %%
df
# %%
df.describe(include='all')
# %%
df['concept:name'].nunique()
# %%

# %%
encoded_data.shape
# %%
a = where(preprocessed_data.num_cols.nunique() > 2
# %%
len(preprocessed_data.num_cols.columns)
# %%
for column in preprocessed_data.num_cols.columns:
    if preprocessed_data.num_cols[column].nunique() <= 2:
        preprocessed_data.num_cols.drop(column, axis=1, inplace=True)
# %%
preprocessed_data.num_cols
# %%
