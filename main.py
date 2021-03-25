# %%
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from preprocessing.encoding import encoder
from preprocessing.preprocessing import preprocessor
from preprocessing.preprocessing import datamanager
from preprocessing.enrich import enrich
from compliance_analysis.aad_experiment.aad.anomaly import detect_anomalies
from compliance_analysis.aad_experiment.aad.anomaly import show_anomaly
tqdm.pandas()
data_source = '/workspaces/thesis/data/preprocessed/2012_A.gzip'
# %%
# Preprocess data
preprocessed = preprocessor.column_rename(
    data_path = data_source,
    case_id_col = 'case:concept:name', activity_col = 'concept:name', 
    timestamp_col = 'time:timestamp', resource_col = 'org:resource')

# %%
# Enrich data
# preprocessed.data = preprocessed.data.groupby(['case:concept:name']).progress_apply(enrich.feature_engineering)
# preprocessed.data = preprocessed.data.groupby(['case:concept:name']).progress_apply(enrich.bounded_existence, activity = 'A_SUBMITTED')

# preprocessed = datamanager(data = preprocessed.data)


# %%
# Encode data
numeric_encoding = {
    'AMOUNT_REQ':'max'}#,
    # 'FirstWithdrawalAmount':'sum',
    # 'NumberOfTerms':'max',
    # 'Accepted':'sum',
    # 'MonthlyCost':'max',
    # 'Selected':'sum',
    # 'CreditScore':'max',
    # 'OfferedAmount':'max',
    # 'case_length':'last',
    # 'activity_count':'last',
    # 'bounded_existence_O_Created':'last'}

# encoded_numeric = encoder.numeric_encoder(preprocessed, numeric_encoding)
# encoded_categorical = encoder.categorical_encoder(preprocessed)
# encoded_data = pd.concat([encoded_numeric, encoded_categorical.reset_index()], axis=1).fillna(0)
encoded_data = pd.read_csv('/workspaces/thesis/data/encoded/encoded_2012.csv')
# preprocessed.num_cols.drop(['Unnamed: 0'], axis=1, inplace=True)
encoded_data.drop(['Unnamed: 0'], axis=1, inplace=True)
# %%
# Detect anomalies 
# aad = detect_anomalies(np.asarray(encoded_data), preprocessed)

# %%
df2 = encoded_data
from sklearn.ensemble import IsolationForest
model=IsolationForest(random_state=0)
model.fit(df2)
df2["scores"] = model.decision_function(df2)
df2["@@index"] = df2.index
df2 = df2[["scores", "@@index"]]
df2 = df2.sort_values("scores")
print(df2)

process = 1
for i in df2['@@index'][7500:7600]:
    show_anomaly(i, preprocessed)
    print('Processed {} of 100..'.format(process))
    process += 1
print('Done')

# %%
preprocessed.data
# %%
import pandas as pd
df = pd.read_parquet('/workspaces/thesis/data/raw/BPI_Challenge_2012.gzip')
# %%
df
# %%
df['org:resource'].unique(ascending=True)
# %%
df.describe(include='all')
# %%
encoded_data.shape
# %%
preprocessed.data['lifecycle:transition'].unique()
# %%
