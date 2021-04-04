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

data_path = '/workspaces/thesis/data/preprocessed/2012_O.gzip'
# %%
# Preprocess data
# preprocessed = preprocessor.column_rename(
    # data_path = data_path,
    # case_id_col = 'case:concept:name', activity_col = 'concept:name', 
    # timestamp_col = 'time:timestamp', resource_col = 'org:resource')

# %%
# Enrich data
# preprocessed.data.rename(columns= {'AMOUNT_REQ':'Request loan amount',
                                #    'activity':'concept:name'}, inplace=True)
# preprocessed.data = preprocessed.data.groupby(['case:concept:name']).filter(lambda g: any(g['concept:name'] == 'O_ACCEPTED'))
# preprocessed.data = preprocessed.data.groupby(['case:concept:name']).progress_apply(enrich.feature_engineering)
# preprocessed.data = preprocessed.data.groupby(['case:concept:name']).progress_apply(enrich.bounded_existence, activity = 'O_ACCEPTED')
# preprocessed.data = preprocessed.data.groupby(['case:concept:name']).progress_apply(enrich.four_eye_principle, activity1 = 'O_CREATED', activity2 = 'O_ACCEPTED')

# %%
# preprocessed = datamanager(data = preprocessed.data)
# preprocessed.num_cols.drop(['Unnamed: 0'], axis=1, inplace=True)
# preprocessed.data.drop(['Unnamed: 0'], axis=1, inplace=True)
# preprocessed.data.to_csv('/workspaces/thesis/data/preprocessed/preprocessed_2012.csv')
# %%
preprocessed = pd.read_csv('/workspaces/thesis/data/preprocessed/preprocessed_2012.csv')
preprocessed = datamanager(data = preprocessed)
# %%
# Encode data
# numeric_encoding = {
    # 'Request loan amount':'max'}#,
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
# encoded_data.to_csv('/workspaces/thesis/data/encoded/encoded_2012.csv')
encoded_data = pd.read_csv('/workspaces/thesis/data/encoded/encoded_2012.csv')
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

# %%
process = 1
for i in df2['@@index'].head(35):
    show_anomaly(i, preprocessed, 'head')
    print('Processed {} of 100..'.format(process))
    process += 1
print('Done')

process = 1
for i in df2['@@index'][36:100]:
    show_anomaly(i, preprocessed, 'head')
    print('Processed {} of 100..'.format(process))
    process += 1
print('Done')

process = 1
for i in df2['@@index'].tail(100):
    show_anomaly(i, preprocessed, 'tail')
    print('Processed {} of 100..'.format(process))
    process += 1
print('Done')

# %%
encoded_data
# %%
encoded_data
# %%
preprocessed.data
# %%
