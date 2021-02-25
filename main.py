# %%
# Imports
import time
from preprocessing.preprocessing import preprocessing
from pm4py.objects.log.util import get_log_representation
from compliance_analysis.aad_experiment.aad.anomaly import detect_anomalies

# %%
# Preprocess data #REWORK TO ONLY EXTRACT THE QUERIED CASE FROM LOG TO VISUALIZE (SKIPS FULL LOG LOADING)
preprocessed_data, log = preprocessing.column_rename(
    data_path = '/workspaces/thesis/data/preprocessed/2017_O.gzip',
    log_path = '/workspaces/thesis/data/raw/BPI Challenge 2017.xes.gz',
    case_id_col = 'case:concept:name', activity_col = 'concept:name', 
    timestamp_col = 'time:timestamp', resource_col = 'org:resource')

# %%
# Encode data #REWORK THIS TO INCLUDE ACTUAL ENCODING..
start = time.time()
encoded_data, feature_names = get_log_representation.get_representation(
    log, 
    str_ev_attr=["concept:name", "org:resource"],
    str_tr_attr=[],#list(preprocessed_data.str_cols.columns), 
    num_ev_attr=[], 
    num_tr_attr=[],#list(preprocessed_data.static_num_cols.columns), 
    str_evsucc_attr=[])#"concept:name", "org:resource"])
end = time.time()
print("Encoding took", end - start, "seconds.")

# %%
# Detect anomalies 
aad = detect_anomalies(encoded_data, log, preprocessed_data.data)






























# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

for i in range (1,100):
    queried=i
    df=preprocessed_data.data
    gp = df.groupby('case:concept:name')
    sample = gp.get_group(df['case:concept:name'].iloc[queried])
    samplee = sample
    sample = sample[['concept:name','org:resource']].value_counts()
    sample = pd.DataFrame(sample)
    sample = sample.reset_index()
    sample.rename(columns={0:'count',
        'org:resource':'resource',
        'concept:name':'activity'
        }, inplace=True)
    dummies = pd.get_dummies(sample['resource'])
    sample = pd.concat([sample['activity'], dummies], axis=1)
    sample = pd.DataFrame(sample.value_counts())
    sample = sample.reset_index()
    sample.rename(columns={0:'count'
        }, inplace=True)
    user_columns = sample.iloc[:,1:-1]
    table_input = sample['activity'].drop_duplicates().reset_index(drop=True)
    for user in user_columns.columns:
        user_data = sample.groupby(['activity'], as_index=False).agg({user:'sum'})
        table_input = pd.concat([table_input, user_data.iloc[:,1:]], axis=1)

    a = np.asarray(table_input.iloc[:,1:])
    vegetables = list(samplee['concept:name'].unique())
    farmers = list(samplee['org:resource'].unique())

    harvest = a

    fig, ax = plt.subplots()
    im = ax.imshow(harvest, cmap='Blues')

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(farmers)))
    ax.set_yticks(np.arange(len(vegetables)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(farmers)
    ax.set_yticklabels(vegetables)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(vegetables)):
        for j in range(len(farmers)):
            text = ax.text(j, i, harvest[i, j],
                        ha="center", va="center", color="w")

    ax.set_title("Harvest of local farmers (in tons/year)")
    fig.tight_layout()
    plt.show()

    # %%
