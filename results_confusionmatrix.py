# %%
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score, recall_score, precision_score
import pandas as pd
import numpy as np
from random import randint
import warnings
warnings.filterwarnings('ignore')




df_survey = pd.read_csv('/workspaces/thesis/AAD Evaluation_May 13, 2021_01.45.csv')
df_survey = df_survey.iloc[2:]
df_survey.drop(columns=['Reflection 2 - Topics', 'Reflection 2 - Parent Topics'], inplace=True)
df_survey_updated = pd.read_csv('/workspaces/thesis/AAD Evaluation - Updated Parameters_May 13, 2021_01.45.csv')
df_survey_updated = df_survey_updated.iloc[2:]
df_survey = pd.DataFrame(np.concatenate([df_survey.values, df_survey_updated.values]), columns=df_survey_updated.columns)

df_survey['StartDate'] = pd.to_datetime(df_survey['StartDate'])
# df_survey = df_survey[df_survey['StartDate'] < '2021-04-22'] # First
# df_survey = df_survey[df_survey['StartDate'].between('2020-04-17','2021-05-01')] # Second
# df_survey = df_survey[df_survey['StartDate'] > '2021-05-01'] # Third

# print(len(df_survey))
print(df_survey['Reflection 2'].value_counts())


numbers = ('1', '2', '3', '4', '5', '6', '7', '8', '9')
select_col = [col for col in df_survey if col.startswith(numbers)]
temp_df_survey = df_survey[select_col].loc[2:]
df_survey = pd.DataFrame(temp_df_survey.astype(float).mean(axis=0))
df_survey['count'] = pd.DataFrame(temp_df_survey.astype(float).count(axis=0))
df_survey = df_survey.reset_index(drop=True)

q1 = df_survey.iloc[:20]
q2 = df_survey.iloc[20:40]
q3 = df_survey.iloc[40:120]
q4 = df_survey.iloc[120:]

q1['Similar'] = (q1[0]*q1['count']-q1['count'])*-1
q2['Similar'] = q2[0]*q2['count']
q3['Similar'] = (q3[0]*q3['count']-q3['count'])*-1
q4['Similar'] = q4[0]*q4['count']

q1['Different'] = q1['count']-q1['Similar']
q2['Different'] = q2['count']-q2['Similar']
q3['Different'] = q3['count']-q3['Similar']
q4['Different'] = q4['count']-q4['Similar']

q1['Question'] = '1'
q2['Question'] = '2'
q3['Question'] = '3'
q4['Question'] = '4'

data = q1, q2, q3, q4
df = pd.DataFrame()

for question in data:
    df = df.append(question)
# %%
print(df[ (df.Question == '1') | (df.Question == '3')]['Similar'].sum())
print(df[ (df.Question == '1') | (df.Question == '3')]['Different'].sum())
print(df[ (df.Question == '2') | (df.Question == '4')]['Similar'].sum())
print(df[ (df.Question == '2') | (df.Question == '4')]['Different'].sum())
# %%
126/(83+126)
# %%
scores = df[0]
scores = scores.mask(scores < 0.5, 0)
scores = scores.mask(scores > 0.5, 1)
scores = scores.mask(scores == 0.5, randint(0,1))

scores['label'] = pd.DataFrame(scores)
scores = scores['label']
scores['pred'] = 1
scores['pred'].iloc[:20] = 0
scores['pred'].iloc[20:40] = 1
scores['pred'].iloc[40:120] = 0
scores['pred'].iloc[120:] = 1

# %%

scores = scores.dropna()

y_true = scores[0]
y_pred = scores['pred']

matrix = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(matrix)
disp.plot(colorbar=False, cmap='Blues')
plt.rcParams.update({'font.size': 14})
# plt.savefig('Confusion_matrix_TOTAL.pdf', transparent=True, dpi=300)

print('F1_score: ' + str(f1_score(y_true, y_pred, average='binary'))[:6])
print('Precision_score: ' + str(precision_score(y_true, y_pred))[:4])
print('Recall_score: ' + str(recall_score(y_true, y_pred))[:4])
print('Accuracy_score: ' + str(accuracy_score(y_true, y_pred))[:4])

# %%
metrics = pd.read_csv('/workspaces/thesis/DEZEMOETJEVOELEN.csv', sep=';')
metrics.drop(columns='session', inplace=True)
metrics = metrics[['TPR', 'F1', 'TNR', 'FPR', 'FNR']]
print(metrics['F1'].mean())
fig, ax = plt.subplots()
fig.set_size_inches(10.5, 6, forward=True)

ax.plot(metrics.TPR, color='#191970')
ax.plot(metrics.F1, color='#0000CD')
ax.plot(metrics.TNR, color='#4169E1')
ax.plot(metrics.FPR, color='#778899')
ax.plot(metrics.FNR, color='#B0C4DE')

ax.legend(metrics.columns,loc='center right')
plt.xticks([0, 1, 2],[1, 2, 3])
ax.yaxis.grid(True, color='gainsboro')

ax.set_xlabel('Evaluation cycle')
ax.set_ylabel('Percentage')

plt.savefig('confusion matrix metrics.pdf', transparent=True, dpi=300)
plt.show()
# %%
