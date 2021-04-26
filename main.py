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
#     data_path = data_path,
#     case_id_col = 'case:concept:name', activity_col = 'concept:name', 
#     timestamp_col = 'time:timestamp', resource_col = 'org:resource')

# %%
# Enrich data
# preprocessed.data.rename(columns= {'AMOUNT_REQ':'Request loan amount',
#                                    'activity':'concept:name'}, inplace=True)
# preprocessed.data = preprocessed.data.groupby(['case:concept:name']).filter(lambda g: any(g['concept:name'] == 'O_ACCEPTED'))
# preprocessed.data = preprocessed.data.groupby(['case:concept:name']).progress_apply(enrich.feature_engineering)
# preprocessed.data = preprocessed.data.groupby(['case:concept:name']).progress_apply(enrich.get_average)
# preprocessed.data = preprocessed.data.groupby(['case:concept:name']).progress_apply(enrich.bounded_existence, activity = 'O_ACCEPTED')
# preprocessed.data = preprocessed.data.groupby(['case:concept:name']).progress_apply(enrich.four_eye_principle, activity1 = 'O_CREATED', activity2 = 'O_ACCEPTED')
# preprocessed.data['concept:name'].replace({'O_SELECTED':'Loan requested',
#                                            'O_CREATED':'Create loan offer',
#                                            'O_SENT':'Sent loan offer to client',
#                                            'O_SENT_BACK':'Receive documents from client',
#                                            'O_ACCEPTED':'Approve loan request',
#                                            'O_CANCELLED':'Cancel loan request'}, inplace=True)
# %%
# preprocessed = datamanager(data = preprocessed.data)
# preprocessed.num_cols.drop(['Unnamed: 0'], axis=1, inplace=True)
# preprocessed.data.drop(['Unnamed: 0'], axis=1, inplace=True)
# preprocessed.data.to_csv('/workspaces/thesis/data/preprocessed/preprocessed_2012.csv')

# %%
# preprocessed = pd.read_csv('/workspaces/thesis/data/preprocessed/preprocessed_2012.csv')
# preprocessed = datamanager(data = preprocessed)

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
df2 = encoded_data
from sklearn.ensemble import IsolationForest
model=IsolationForest(random_state=0)
model.fit(df2)
df2["scores"] = model.decision_function(df2)
df2["@@index"] = df2.index
df2 = df2[["scores", "@@index"]]
df2 = df2.sort_values("scores")
df_survey = pd.read_csv('/workspaces/thesis/AAD Evaluation_April 23, 2021_02.54.csv')
numbers = ('1', '2', '3', '4', '5', '6', '7', '8', '9')
select_col = [col for col in df_survey if col.startswith(numbers)]
df_survey = df_survey[select_col].loc[2:]
df_survey = pd.DataFrame(df_survey.astype(float).mean(axis=0))#.transpose().dropna(axis=1)
df_survey = df_survey.reset_index(drop=True)
df_one = df2['@@index'].head(100).sort_values()
df_three = df2['@@index'].tail(100).sort_values()
df_one = df_one.append([df_three])
df_one = pd.DataFrame(df_one.reset_index(drop=True))
df_survey['index'] = df_one['@@index'].sort_values()
df_survey = df_survey.set_index(df_survey['index'])
df_survey.fillna(value=0, inplace=True)
query = df_survey.index
df_survey.drop(columns=['index'])
encoded_data['score'] = df_survey[0] 
y = encoded_data['score']
encoded_data.drop('score', axis=1, inplace=True)

# %%
# Detect anomalies 
aad = detect_anomalies(np.asarray(encoded_data), y, query)

# %%
# process = 1
# for i in df2['@@index'].head(35):
#     show_anomaly(i, preprocessed, 'head')
#     print('Processed {} of 100..'.format(process))
#     process += 1

# for i in df2['@@index'][36:100]:
#     show_anomaly(i, preprocessed, 'head')
#     print('Processed {} of 100..'.format(process))
#     process += 1
# print('Done')

# process = 1
# for i in df2['@@index'].tail(100):
#     show_anomaly(i, preprocessed, 'tail')
#     print('Processed {} of 100..'.format(process))
#     process += 1
# print('Done')
