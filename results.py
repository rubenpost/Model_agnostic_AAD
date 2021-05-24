# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

#/workspaces/thesis/AAD Evaluation_May 13, 2021_01.45.csv
#/workspaces/thesis/AAD Evaluation - Updated Parameters_May 13, 2021_01.45.csv
df_survey = pd.read_csv('/workspaces/thesis/Limperg Institute - Symposium Statistical Auditing_May 19, 2021_14.25.csv')
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

    # Comment this code for second evaluation auditors dat

df_survey['StartDate'] = pd.to_datetime(df_survey['StartDate'])
# df_survey = df_survey[df_survey['StartDate'] < '2021-04-18']

numbers = ('1', '2', '3', '4', '5', '6', '7', '8', '9')
select_col = [col for col in df_survey if col.startswith(numbers)]
temp_df_survey = df_survey[select_col].loc[2:]
df_survey = pd.DataFrame(temp_df_survey.astype(float).mean(axis=0))
df_survey['count'] = pd.DataFrame(temp_df_survey.astype(float).count(axis=0))
df_survey = df_survey.reset_index(drop=True)

q1 = df_survey.iloc[:20]
q2 = df_survey.iloc[20:40]
q3 = df_survey.iloc[40:80]
q4 = df_survey.iloc[80:120]
q5 = df_survey.iloc[120:160]
q6 = df_survey.iloc[160:200]

q1['Similar'] = (q1[0]*q1['count']-q1['count'])*-1
q2['Similar'] = q2[0]*q2['count']
q3['Similar'] = (q3[0]*q3['count']-q3['count'])*-1
q4['Similar'] = (q4[0]*q4['count']-q4['count'])*-1
q5['Similar'] = q5[0]*q5['count']
q6['Similar'] = q6[0]*q6['count']


q1['Different'] = q1['count']-q1['Similar']
q2['Different'] = q2['count']-q2['Similar']
q3['Different'] = q3['count']-q3['Similar']
q4['Different'] = q4['count']-q4['Similar']
q5['Different'] = q5['count']-q5['Similar']
q6['Different'] = q6['count']-q6['Similar']

q1['Question'] = '1'
q2['Question'] = '2'
q3['Question'] = '3'
q4['Question'] = '4'
q5['Question'] = '5'
q6['Question'] = '6'

data = q1, q2, q3, q4, q5, q6
df = pd.DataFrame()

for question in data:
    df = df.append(question)
 
df = df.groupby(['Question']).agg({'Similar':'sum','Different':'sum'})

labels = df.index
Similar = df['Similar']
different = df['Different']
width = 0.5

fig, ax = plt.subplots()

ax.bar(labels, Similar, width, label='Similar', color='#FFDB00')
ax.bar(labels, different, width, bottom=Similar, label='Different', color='#cccccc')
ax.legend(frameon=True)
from matplotlib.font_manager import FontProperties

fontP = FontProperties()
fontP.set_size('small')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)

ax.set_xlabel('Question')
ax.set_title(' ')

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

plt.savefig('test_angelique.png', transparent=True, dpi=300, bbox_inches='tight')
plt.show()

low_sim = df.iloc[0][0] + df.iloc[2][0] + df.iloc[3][0]
low_diff = df.iloc[0][1] + df.iloc[2][1] + df.iloc[3][1]
print('percentage of low score cases corrent: ', (low_sim/low_diff) / (low_sim+low_diff), '%')

high_sim = df.iloc[1][0] + df.iloc[4][0] + df.iloc[5][0]
high_diff = df.iloc[1][1] + df.iloc[4][1] + df.iloc[5][1]
print('percentage of high score cases corrent: ', (high_sim/high_diff) / (high_sim+high_diff), '%')

# %%
