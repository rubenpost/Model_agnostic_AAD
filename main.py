# %%
# Imports
import time
import pandas as pd
import numpy as np
import importlib as imp
from tqdm.auto import tqdm
from preprocessing import encoding
from preprocessing.preprocessing import preprocessor
from compliance_analysis.compliance_rules.compliance_checker import filter
from compliance_analysis.aad_experiment.aad.anomaly import detect_anomalies
from compliance_analysis.aad_experiment.aad.anomaly import show_anomaly
tqdm.pandas()
data_source = '/workspaces/thesis/data/raw/BPI_Challenge_2017.gzip'
# %%
# Preprocess data
preprocessed_data = preprocessor.column_rename(
    data_path = data_source,
    case_id_col = 'case:concept:name', activity_col = 'concept:name', 
    timestamp_col = 'time:timestamp', resource_col = 'org:resource')

# %%
# Enrich data

# df = preprocessed_data.data.groupby(['case:concept:name']).progress_apply(filter.bounded_existence, activity = 'O_Created')

# %%
# Encode data #REWORK THIS TO INCLUDE ACTUAL ENCODING..
start = time.time()

static_num_cols = pd.concat([preprocessed_data.static_num_cols, preprocessed_data.case_id_col], axis=1)
encoded_snc = static_num_cols.groupby(['case:concept:name'], as_index=False).agg(['max'])
dummies = pd.get_dummies(preprocessed_data.static_cat_cols[['ApplicationType', 'LoanGoal']])
static_cat = pd.concat([preprocessed_data.case_id_col, dummies], axis=1)
encoded_scc = static_cat.groupby(['case:concept:name'], as_index=False).agg(['max'])
static_data = pd.concat([encoded_snc, encoded_scc], axis=1)
dummies = pd.get_dummies(preprocessed_data.dynamic_cat_cols[['Action', 'org:resource', 'concept:name']])
dynamic_cat = pd.concat([preprocessed_data.case_id_col, dummies], axis=1)
encoded_dcc = dynamic_cat.groupby(['case:concept:name'], as_index=False).agg(['max'])
dynamic_data = pd.concat([encoded_dcc, encoded_dcc], axis=1)

encoded_data = pd.concat([dynamic_data, static_data], axis=1)
encoded_data = np.asarray(encoded_data)

end = time.time()
print("Encoding took", end - start, "seconds.")

# %%
# Detect anomalies 
aad = detect_anomalies(encoded_data, preprocessed_data)

# %%
encoded_data
# %%
imp.reload(module=encoding)
encoders = {'RequestedAmount':'Mean'}
a, b = encoding.encoder(data=preprocessed_data.data, encoders=encoders)
# %%
data
# %%
preprocessed_data.data.columns
# %%
df
# %%
preprocessed_data.data['concept:name'].unique()

# %%
df
# %%

data = preprocessed_data.data
gp = data.groupby('case:concept:name')
sample = gp.get_group(data['case:concept:name'].unique()[4]).reset_index()
table_values = pd.DataFrame(columns=['Antecedent', 'Consequent'])
for index in range(0, len(sample)-1):
    table_values.loc[index] = sample['concept:name'].iloc[index], sample['concept:name'].iloc[index+1]
table_values.value_counts().reset_index()
# %%
sample = pd.DataFrame(pd.concat([table_values['Antecedent'], pd.get_dummies(table_values['Consequent'])], axis=1).value_counts()).reset_index()
user_columns = sample.iloc[:,1:-1]
table_input = sample['Antecedent'].drop_duplicates().reset_index(drop=True)
for user in user_columns.columns:
    user_data = sample.groupby(['Antecedent'], as_index=False).agg({user512737
    :'sum'})
    table_input = pd.concat([table_input, user_data.iloc[:,1:]], axis=1)
table_input = np.asarray(table_input.iloc[:,1:])
table_input
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", palette="muted", color_codes=True, font_scale = 1)
fig, ax = plt.subplots(1, figsize=(10, 10))
sns.despine(left=True)
activities, resources = list(table_values['Antecedent'].unique()), list(table_values['Consequent'].unique())
ax.imshow(table_input, cmap='Blues')
# We want to show all ticks...
ax.set_xticks(np.arange(len(resources)))
ax.set_yticks(np.arange(len(activities)))
# ... and label them with the respective list entries
ax.set_xticklabels(resources)
ax.set_yticklabels(activities)
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
        rotation_mode="anchor")
for i in range(len(activities)):
    for j in range(len(resources)):
        text = ax.text(j, i, table_input[i, j],
                    ha="center", va="center", color="w")
# %%
import seaborn as sns
import matplotlib.pyplot as plt
ax = sns.distplot(plooty["RequestedAmount"], norm_hist=False, kde=False, color="orange")
ax.set(ylabel='Count')
# %%
heights = [h.get_height() for h in patches]
max(heights)/10048
# %%
import math
sns.set(style="white", palette="muted", color_codes=True, font_scale = 1)
cmap = plt.get_cmap('jet')
fig, ax = plt.subplots(1, figsize=(10, 10))
ax=sns.distplot(plooty["RequestedAmount"], norm_hist=False, kde=False, color="orange")
patches = ax.patches
patches[1].get_height()
# %%
bin_width = patches[0].get_width()
bin_number = math.ceil(20000 / bin_width)
patches[bin_number].set_facecolor(cmap(0,5))
ax.set(ylabel='Count')
ax.axvline(x=bin_width*bin_number, color=cmap(0,5), ymax=0)
ax.legend(labels=['Sample','Population'])
variable_value = '20000'
text = "    \u27f5  "  + variable_value
plt.text(20000, 1240, text)
plt.show()
# %%

# %%
for case in preprocessed_data.data['case:concept:name']: print(case)#preprocessed_data.num_cols["RequestedAmount"]
# %%
preprocessed_data.data.dropna().groupby(['case:concept:name']).max().reset_index()
# %%
preprocessed_data.data.dropna(subset=['RequestedAmount'])
# %%
preprocessed_data.data.isna().sum()
# %%
data = pd.concat([preprocessed_data.num_cols, preprocessed_data.case_id_col], axis=1)
plooty = data.dropna(subset=preprocessed_data.num_cols.columns).groupby(['case:concept:name']).max().reset_index()
# %%
plooty['RequestedAmount'].value_counts()
# %%
np.histogram
# %%
for i in range (0,19): print(patches[i].get_height())
# %%
round(2.7)
# %%
preprocessed_data.data.sort_values(by='concept:name')
# %%
# %%
# Imports
import time
import pandas as pd
import numpy as np
import importlib as imp
from tqdm.auto import tqdm
from preprocessing import encoding
from preprocessing.preprocessing import preprocessor
from compliance_analysis.compliance_rules.compliance_checker import filter
from compliance_analysis.aad_experiment.aad.anomaly import detect_anomalies
from compliance_analysis.aad_experiment.aad.anomaly import show_anomaly
tqdm.pandas()
data_source = '/workspaces/thesis/data/raw/BPI_Challenge_2017.gzip'
# Preprocess data
preprocessed_data = preprocessor.column_rename(
    data_path = data_source,
    case_id_col = 'case:concept:name', activity_col = 'concept:name', 
    timestamp_col = 'time:timestamp', resource_col = 'org:resource')
show_anomaly(5, preprocessed_data)
# %%
