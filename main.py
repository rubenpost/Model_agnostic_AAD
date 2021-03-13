# %%
import numpy as np
from tqdm.auto import tqdm
from preprocessing.encoding import encoder
from preprocessing.preprocessing import preprocessor
from preprocessing.preprocessing import datamanager
from preprocessing.enrich import enrich
from compliance_analysis.aad_experiment.aad.anomaly import detect_anomalies
tqdm.pandas()
data_source = '/workspaces/thesis/data/enriched/2017_with_features.gzip'
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
import pandas as pd
df = preprocessed
gp = df.data.groupby('case:concept:name')
case_activities = gp.get_group(df.data['case:concept:name'].unique()[0])
case_activities = pd.DataFrame(case_activities['concept:name'].value_counts()).reset_index()
case_activities = case_activities.pivot_table(columns='index', values='concept:name')
# %%
case_activities
# %%
preprocessed.data = preprocessed.data[:100:]
# %%
import pandas as pd
def object_encoder(self, preprocessed):
    for columns in self.drop(['EventID'], axis=1).columns:
        encoded_objects = self[columns].value_counts().reset_index()
        encoded_objects = encoded_objects.pivot_table(columns='index', values=columns)
    for column in preprocessed.object_cols.columns:
        encoded_objects[column] = preprocessed.data[column]
    return encoded_objects

encoded_objects = preprocessed.object_cols.head(1000).groupby(['case:concept:name']).progress_apply(object_encoder, preprocessed = preprocessed)
# %%
encoded_objects
# %%
preprocessed.object_cols.columns
# %%
