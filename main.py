# %%
import time
import pandas as pd
import numpy as np
import importlib as imp
from tqdm.auto import tqdm
from preprocessing.encoding import encoder
from preprocessing.preprocessing import preprocessor
from preprocessing.preprocessing import datamanager
from compliance_analysis.compliance_rules.compliance_checker import enrich
from compliance_analysis.aad_experiment.aad.anomaly import detect_anomalies
tqdm.pandas()
data_source = '/workspaces/thesis/2017_with_features.gzip'
# %%
# Preprocess data
preprocessed = preprocessor.column_rename(
    data_path = data_source,
    case_id_col = 'case:concept:name', activity_col = 'concept:name', 
    timestamp_col = 'time:timestamp', resource_col = 'org:resource')

# %%
# Enrich data
# preprocessed.data = preprocessed.data.groupby(['case:concept:name']).progress_apply(enrich.feature_engineering)
# preprocessed.data = preprocessed.data.groupby(['case:concept:name']).progress_apply(enrich.bounded_existence, activity = 'O_Created')

# %%
# Encode data
numeric_encoding = {
    'RequestedAmount':'max',
    'FirstWithdrawalAmount':'sum',
    'NumberOfTerms':'max',
    'Accepted':'sum',
    'MonthlyCost':'max',
    'Selected':'sum',
    'CreditScore':'max',
    'OfferedAmount':'max',
    'case_length':'last',
    'activity_count':'last',
    'bounded_existence_O_Created':'last'}

# encoded_cat = encoder.categorical_encoder(preprocessed)

encoded_numeric = encoder.numeric_encoder(preprocessed.data, numeric_encoding)
# encoded_data = pd.concat([encoded_numeric, encoded_cat], axis=1).fillna('0').drop(['case:concept:name'], axis=1)

# %%
# Detect anomalies 
aad = detect_anomalies(np.asarray(encoded_numeric), preprocessed)

# %%
preprocessed.num_cols.isna().sum()

# %
gp = preprocessed.data.groupby('case:concept:name')
queried_case = gp.get_group(preprocessed.data['case:concept:name'].unique()[17370])

# %%
queried_case
# %%
encoded_numeric.isna().sum()
# %%
