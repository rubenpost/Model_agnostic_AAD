# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


df_survey = pd.read_csv('/workspaces/thesis/AAD Evaluation - Updated Parameters_May 12, 2021_02.27.csv')
df_survey = df_survey.iloc[2:]

def cleanup():
    pass
    # Uncomment thuis code for second evaluation auditors data
    # df_survey = pd.read_csv('/workspaces/thesis/AAD Evaluation_May 3, 2021_08.57.csv')
    # raw = df_survey
    # raw = raw.iloc[2:]
    # one = pd.DataFrame(raw.iloc[:3])
    # two = pd.DataFrame(raw.iloc[40:40])
    # two = two.append(one)
    # df_survey = two

    # Comment this code for second evaluation auditors data
    # df_survey['StartDate'] = pd.to_datetime(df_survey['StartDate'])
    # df_survey = df_survey[df_survey['StartDate'] > '2021-04-28']

numbers = ('1', '2', '3', '4', '5', '6', '7', '8', '9')
select_col = [col for col in df_survey if col.startswith(numbers)]
temp_df_survey = df_survey[select_col].loc[2:]
df_survey = pd.DataFrame(temp_df_survey.astype(float).mean(axis=0))
df_survey['count'] = pd.DataFrame(temp_df_survey.astype(float).count(axis=0))
df_survey = df_survey.reset_index(drop=True)

q1 = df_survey.iloc[:20]
q2 = df_survey.iloc[20:40]
q3 = df_survey.iloc[40:120]
q4 = df_survey.iloc[120:200]

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
 
df = df.groupby(['Question']).agg({'Similar':'sum','Different':'sum'})

labels = df.index
Similar = df['Similar']
different = df['Different']
width = 0.5

fig, ax = plt.subplots()

ax.bar(labels, Similar, width, label='Similar', color='#C8EABB')
ax.bar(labels, different, width, bottom=Similar, label='Different', color='#F6CDC4')
ax.legend(frameon=False)

ax.set_xlabel('Question')
ax.set_title('The students answers compared to our algorithm')

for spine in ax.spines.values():
    spine.set_visible(False)

ax.spines['bottom'].set_visible(True)

ax.axes.yaxis.set_ticks([])

for rect in ax.patches:
    height = rect.get_height()
    width = rect.get_width()
    x = rect.get_x()
    y = rect.get_y()

    label_text = f'{height:.0f}'
    
    label_x = x + width / 2
    label_y = y + height / 2

    if height > 0:
        ax.text(label_x, label_y, label_text, ha='center', va='center', fontsize=8, color='black')

plt.savefig('test_angelique.png', transparent=True, dpi=300)
plt.show()
