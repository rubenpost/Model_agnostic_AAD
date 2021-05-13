
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
#     # 'activity_count':'max',
#     # 'average_cancellation':'max',
#     # 'average_resource':'max',
#     # 'bounded_existence_O_Accepted':'max',
#     # 'Binding_of_duties_O_Created_O_Accepted':'max'}

# # preprocessed.dynamic_cat_cols.drop(columns=['EventID'], axis=1, inplace=True)
# encoded_numeric = encoder.numeric_encoder(preprocessed, numeric_encoding)
# encoded_categorical = encoder.categorical_encoder(preprocessed)
# encoded_data = pd.concat([encoded_numeric, encoded_categorical.reset_index()], axis=1).fillna(0)
# encoded_data.to_csv('/workspaces/thesis/data/encoded/encoded_2012_W.csv')
encoded_data = pd.read_csv('/workspaces/thesis/data/encoded/encoded_2012.csv')

# %%
# Integrate feedback
df2 = encoded_data
df2.drop(columns='case:concept:name', axis=1, inplace=True)
# %%
from sklearn.ensemble import IsolationForest
model=IsolationForest(random_state=0)
model.fit(df2)
df2["scores"] = model.decision_function(df2)
df2["@@index"] = df2.index
df2 = df2[["scores", "@@index"]]
df2 = df2.sort_values("scores")

df_survey = pd.read_csv('/workspaces/thesis/AAD Evaluation_May 3, 2021_08.57.csv')
numbers = ('1', '2', '3', '4', '5', '6', '7', '8', '9')
select_col = [col for col in df_survey if col.startswith(numbers)]
test = df_survey[select_col].loc[2:]
df_survey = pd.DataFrame(test.astype(float).mean(axis=0))#.transpose().dropna(axis=1)
# print(df_survey)
df_survey['count'] = pd.DataFrame(test.astype(float).count(axis=0))#.transpose().dropna(axis=1)
df_survey = df_survey.reset_index(drop=True)
df_one = df2['@@index'].head(100).sort_values()
print(df_one)
df_three = df2['@@index'].tail(100).sort_values()
print(df_three)
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

# %%
# Detect anomalies 
aad = detect_anomalies(np.asarray(encoded_data), y, query)

# %%
# Produce visuals
process = 1
for i in np.asarray(df2['@@index'])[:1]:
    show_anomaly(int(i), preprocessed, encoded_data, 'new_head')
    print('Processed {} of 100..'.format(process))
    process += 1
print('Done')

# process = 1
# for i in np.asarray(df2['@@index'])[-100:]:
#     show_anomaly(int(i), preprocessed, encoded_data, 'new_tail')
#     print('Processed {} of 100..'.format(process))
#     process += 1
# print('Done')

# %%
preprocessed.data[preprocessed.data['activity_count'] >= 161]['case:concept:name'].nunique()
preprocessed.data['case:concept:name'].nunique()
# %%
1917/2243

# %%
df_survey.to_csv('test.csv')
# %%
preprocessed.data.loc[9]

# %%            
df_survey = pd.read_csv('/workspaces/thesis/AAD Evaluation_May 3, 2021_08.57.csv')
raw = df_survey
numbers = ('1', '2', '3', '4', '5', '6', '7', '8', '9')
select_col = [col for col in df_survey if col.startswith(numbers)]
test = df_survey[select_col].loc[2:]
df_survey = pd.DataFrame(test.astype(float).mean(axis=0))#.transpose().dropna(axis=1)
df_survey['count'] = pd.DataFrame(test.astype(float).count(axis=0))#.transpose().dropna(axis=1)
df_survey = df_survey.reset_index(drop=True)

q1 = df_survey.iloc[:20]
q2 = df_survey.iloc[20:40]
q3 = df_survey.iloc[40:120]
q4 = df_survey.iloc[120:200]

q1['correct'] = (q1[0]*q1['count']-q1['count'])*-1
q2['correct'] = q2[0]*q2['count']
q3['correct'] = (q3[0]*q3['count']-q3['count'])*-1
q4['correct'] = q4[0]*q4['count']

q1['incorrect'] = q1['count']-q1['correct']
q2['incorrect'] = q2['count']-q2['correct']
q3['incorrect'] = q3['count']-q3['correct']
q4['incorrect'] = q4['count']-q4['correct']

q1['question'] = 'q1'
q2['question'] = 'q2'
q3['question'] = 'q3'
q4['question'] = 'q4'

data = q1, q2, q3, q4
df = pd.DataFrame()

for i in data:
    df = df.append(i)
# %%
q1.to_csv('q1.csv', index=False, encoding='utf-8')
q2.to_csv('q2.csv', index=False)
q3.to_csv('q3.csv', index=False)
q4.to_csv('q4.csv', index=False)
# %%
df_survey = pd.read_csv('/workspaces/thesis/AAD Evaluation - Updated Parameters_May 7, 2021_02.46.csv')
raw = df_survey
raw = raw.iloc[2:]
raw['StartDate'] = pd.to_datetime(raw['StartDate'])
# raw = raw[raw['StartDate'] > '2021-04-29']
df_survey = raw[raw['StartDate'] > '2021-04-28']
print(df_survey)
numbers = ('1', '2', '3', '4', '5', '6', '7', '8', '9')
select_col = [col for col in df_survey if col.startswith(numbers)]
test = df_survey[select_col].loc[2:]
df_survey = pd.DataFrame(test.astype(float).mean(axis=0))#.transpose().dropna(axis=1)
df_survey['count'] = pd.DataFrame(test.astype(float).count(axis=0))#.transpose().dropna(axis=1)
df_survey = df_survey.reset_index(drop=True)

q1 = df_survey.iloc[:20]
q2 = df_survey.iloc[20:40]
q3 = df_survey.iloc[40:120]
q4 = df_survey.iloc[120:200]

q1['correct'] = (q1[0]*q1['count']-q1['count'])*-1
q2['correct'] = q2[0]*q2['count']
q3['correct'] = (q3[0]*q3['count']-q3['count'])*-1
q4['correct'] = q4[0]*q4['count']

q1['incorrect'] = q1['count']-q1['correct']
q2['incorrect'] = q2['count']-q2['correct']
q3['incorrect'] = q3['count']-q3['correct']
q4['incorrect'] = q4['count']-q4['correct']

q1['question'] = '1'
q2['question'] = '2'
q3['question'] = '3'
q4['question'] = '4'

data = q1, q2, q3, q4
df = pd.DataFrame()

for i in data:
    df = df.append(i)
# %%
df.to_csv('evaluation_two.csv', encoding='utf-8')
# %%
df_survey = pd.read_csv('/workspaces/thesis/AAD Evaluation_May 3, 2021_08.57.csv')
raw = df_survey
raw = raw.iloc[2:]
one = pd.DataFrame(raw.iloc[-1:])
two = pd.DataFrame(raw.iloc[39:40])
two = two.append(one)
df_survey = two
numbers = ('1', '2', '3', '4', '5', '6', '7', '8', '9')
select_col = [col for col in df_survey if col.startswith(numbers)]
test = df_survey[select_col].loc[2:]
df_survey = pd.DataFrame(test.astype(float).mean(axis=0))#.transpose().dropna(axis=1)
df_survey['count'] = pd.DataFrame(test.astype(float).count(axis=0))#.transpose().dropna(axis=1)
df_survey = df_survey.reset_index(drop=True)

q1 = df_survey.iloc[:20]
q2 = df_survey.iloc[20:40]
q3 = df_survey.iloc[40:120]
q4 = df_survey.iloc[120:200]

q1['correct'] = (q1[0]*q1['count']-q1['count'])*-1
q2['correct'] = q2[0]*q2['count']
q3['correct'] = (q3[0]*q3['count']-q3['count'])*-1
q4['correct'] = q4[0]*q4['count']

q1['incorrect'] = q1['count']-q1['correct']
q2['incorrect'] = q2['count']-q2['correct']
q3['incorrect'] = q3['count']-q3['correct']
q4['incorrect'] = q4['count']-q4['correct']

q1['question'] = '1'
q2['question'] = '2'
q3['question'] = '3'
q4['question'] = '4'

data = q1, q2, q3, q4
df = pd.DataFrame()

for i in data:
    df = df.append(i)
# %%
df
# %%
df.to_csv('evaluation_three.csv')
# %%
df = df.groupby(['question']).agg({'correct':'sum','incorrect':'sum'})
import matplotlib.pyplot as plt
df
# %%

colors = ["#FFDB00", "#797878"]
df.plot.bar(stacked=True, color=colors, figsize=(10,7))