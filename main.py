
# %%
import numpy as np
import pandas as pd
from random import randint
from tqdm.auto import tqdm
from preprocessing.encoding import encoder
from preprocessing.preprocessing import preprocessor
from preprocessing.preprocessing import datamanager
from preprocessing.enrich import enrich
from compliance_analysis.aad_experiment.aad.anomaly import detect_anomalies
from compliance_analysis.aad_experiment.aad.anomaly import show_anomaly
tqdm.pandas()

data_path = '/workspaces/thesis/data/raw/BPI_Challenge_2012.gzip'

# Preprocess data
preprocessed = preprocessor.column_rename(
    data_path = data_path,
    case_id_col = 'case:concept:name', activity_col = 'concept:name', 
    timestamp_col = 'time:timestamp', resource_col = 'org:resource')

# Enrich data
preprocessed.data.rename(columns= {'AMOUNT_REQ':'Request loan amount',
                                   'activity':'concept:name'}, inplace=True)
preprocessed.data = preprocessed.data.groupby(['case:concept:name']).filter(lambda g: any(g['concept:name'] == 'O_ACCEPTED'))
preprocessed.data = preprocessed.data.groupby(['case:concept:name']).progress_apply(enrich.feature_engineering)
preprocessed.data = preprocessed.data.groupby(['case:concept:name']).progress_apply(enrich.get_average, activity = 'O_CANCELLED')
preprocessed.data = preprocessed.data.groupby(['case:concept:name']).progress_apply(enrich.bounded_existence, activity = 'W_Beoordelen fraude', count = 0)
preprocessed.data = preprocessed.data.groupby(['case:concept:name']).progress_apply(enrich.binding_duties, activity1 = 'O_CREATED', activity2 = 'O_ACCEPTED')
preprocessed.data = preprocessed.data[(preprocessed.data['concept:name'].str[:1] == 'O')]# | (preprocessed.data['concept:name'].str[:1] == 'O')]
preprocessed.data['concept:name'] = preprocessed.data['concept:name'].str[2:]
preprocessed.data['concept:name'].replace({'SELECTED':'Loan requested',
                                            'CREATED':'Create loan offer',
                                            'SENT':'Sent loan offer to client',
                                            'SENT_BACK':'Receive documents from client',
                                            'ACCEPTED':'Approve loan request',
                                            'CANCELLED':'Cancel loan request'}, inplace=True)

preprocessed = datamanager(data = preprocessed.data)
preprocessed.num_cols.drop(['Unnamed: 0'], axis=1, inplace=True)
preprocessed.data.drop(['Unnamed: 0'], axis=1, inplace=True)
preprocessed.data.to_csv('/workspaces/thesis/data/preprocessed/preprocessed_2012_W.csv')

# %%
# Encode data
# numeric_encoding = {
#     'Request loan amount':'last',
#     'Case length in calendar days':'max',
#     'activity_count':'max',
#     'average_cancellation':'max',
#     'average_resource':'max',
#     'Binding_of_duties_O_CREATED_O_ACCEPTED':'last',
#     'bounded_existence_W_Beoordelen fraude':'last'}#,
# #     # 'activity_count':'max',
# #     # 'average_cancellation':'max',
# #     # 'average_resource':'max',
# #     # 'bounded_existence_O_Accepted':'max',
# #     # 'Binding_of_duties_O_Created_O_Accepted':'max'}


# encoded_numeric = encoder.numeric_encoder(preprocessed, numeric_encoding)
# encoded_categorical = encoder.categorical_encoder(preprocessed)
# encoded_data = pd.concat([encoded_numeric, encoded_categorical.reset_index()], axis=1).fillna(0)
# encoded_data.to_csv('/workspaces/thesis/data/encoded/encoded_2012_W.csv')
encoded_data = pd.read_csv('/workspaces/thesis/data/encoded/encoded_2012_W.csv')

# %%
# Integrate feedback
df2 = encoded_data
df2.drop(columns='case:concept:name', axis=1, inplace=True)
from sklearn.ensemble import IsolationForest
model=IsolationForest(random_state=0)
model.fit(df2)
df2["scores"] = model.decision_function(df2)
df2["@@index"] = df2.index
df2 = df2[["scores", "@@index"]]
df2 = df2.sort_values("scores")

df_survey = pd.read_csv('/workspaces/thesis/data/survey/AAD Evaluation_May 25, 2021_01.38.csv')
df_survey = df_survey.iloc[2:]
df_survey.drop(columns=['Reflection 2 - Topics', 'Reflection 2 - Parent Topics'], inplace=True)
df_survey_updated = pd.read_csv('/workspaces/thesis/data/survey/AAD Evaluation - Updated Parameters_May 25, 2021_01.37.csv')
df_survey_updated = df_survey_updated.iloc[2:]
df_survey = pd.DataFrame(np.concatenate([df_survey.values, df_survey_updated.values]), columns=df_survey_updated.columns)
numbers = ('1', '2', '3', '4', '5', '6', '7', '8', '9')
select_col = [col for col in df_survey if col.startswith(numbers)]
test = df_survey[select_col].loc[2:]
df_survey = pd.DataFrame(test.astype(float).mean(axis=0))#.transpose().dropna(axis=1)
df_survey['count'] = pd.DataFrame(test.astype(float).count(axis=0))#.transpose().dropna(axis=1)
df_survey = df_survey.reset_index(drop=True)
df_one = df2['@@index'].head(100).sort_values()
df_three = df2['@@index'].tail(100).sort_values()
df_one = df_one.append([df_three])
df_one = pd.DataFrame(df_one.reset_index(drop=True))
df_survey['index'] = df_one['@@index'].sort_values()
data_vis = df_survey
df_survey = df_survey.set_index(df_survey['index'])
df_survey.fillna(value=0, inplace=True)
query = df_survey.index
df_survey.drop(columns=['index'])
encoded_data['score'] = df_survey[0] 
y = encoded_data['score']
encoded_data.drop('score', axis=1, inplace=True)
y = y.mask(y > 0.5, 1)
y = y.mask(y < 0.5, 0)
y[y == 0.5] = randint(0,1)

# %%
# Detect anomalies 
aad = detect_anomalies(np.asarray(encoded_data), y, query)

# %%
# Produce visuals
process = 1
for i in np.asarray(aad[0])[93:101]:
    show_anomaly(int(i), preprocessed, encoded_data, 'limperg_head')
    print('Processed {} of 100..'.format(process))
    process += 1
print('Done')

for i in np.asarray(aad[0])[:92]:
    show_anomaly(int(i), preprocessed, encoded_data, 'limperg_head')
    print('Processed {} of 100..'.format(process))
    process += 1

process = 1
for i in np.asarray(aad[0])[-100:]:
    show_anomaly(int(i), preprocessed, encoded_data, 'limperg_tail')
    print('Processed {} of 100..'.format(process))
    process += 1
print('Done')

# %%

# %%
