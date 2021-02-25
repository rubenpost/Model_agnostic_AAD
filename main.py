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

queried=10
df=preprocessed_data.data
gp = df.groupby('case:concept:name')
sample = gp.get_group(df['case:concept:name'].iloc[queried])

table = sample[['concept:name','org:resource']].value_counts()
table = pd.DataFrame(table)
table = table.reset_index()
table.rename(columns={0:'count',
    'org:resource':'resource',
    'concept:name':'activity'
    }, inplace=True)
dummies = pd.get_dummies(table['resource'])
new_df = pd.concat([table['activity'], dummies], axis=1)
extra_data = new_df.append(new_df)
extra_data.unique
# %%
vegetables = list(sample['concept:name'])
farmers = list(sample['org:resource'])

harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
                    [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                    [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                    [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
                    [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
                    [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])


fig, ax = plt.subplots()
im = ax.imshow(harvest)

# We want to show all ticks...
ax.set_xticks(np.arange(len(farmers)))
ax.set_yticks(np.arange(len(vegetables)))
# ... and label them with the respective list entries
ax.set_xticklabels(farmers)
ax.set_yticklabels(vegetables)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# # Loop over data dimensions and create text annotations.
# for i in range(len(vegetables)):
#     for j in range(len(farmers)):
#         text = ax.text(j, i, harvest[i, j],
#                        ha="center", va="center", color="w")

ax.set_title("Harvest of local farmers (in tons/year)")
fig.tight_layout()
plt.show()

# %%
import pandas as pd
from bokeh.io import show
from bokeh.io.output import output_notebook
from bokeh.models import (BasicTicker, ColorBar, ColumnDataSource,
                          LinearColorMapper, PrintfTickFormatter,)
from bokeh.plotting import figure
from bokeh.transform import transform
from bokeh.palettes import RdBu9
from bokeh.plotting import ColumnDataSource, figure, show

data = preprocessed_data.data
gp = data.groupby('case:concept:name')
sample = gp.get_group('Application_1387439149')
df1 = sample[['org:resource', 'concept:name']].append(sample[['org:resource', 'concept:name']][:6])
df = df1.value_counts()
df = pd.DataFrame(df)
df = df.reset_index()
df.rename(columns={0:'count',
'org:resource':'resource',
'concept:name':'name'}, inplace=True)
df
source = ColumnDataSource(df)
mapper = LinearColorMapper(palette=RdBu9, low=df['count'].min(), high=df['count'].max())
tooltips = [
    ("Resource", "@resource"),
    ("Activity", "@name"),
    ("Count", "@count"),
]
p = figure(plot_width=df['resource'].nunique()*150, plot_height=df['name'].nunique()*100, title="Activities performed per preparer",
           y_range=df['name'].unique(), x_range=df['resource'].unique(),x_axis_location="above", tooltips=tooltips)

p.rect(y="name", x="resource", width=1, height=1, source=source,
       line_color=None, fill_color=transform('count', mapper))

p.axis.axis_line_color = None
p.axis.major_tick_line_color = None
p.axis.major_label_text_font_size = "13px"
p.axis.major_label_standoff = 0
p.xaxis.major_label_orientation = 1.2
p.outline_line_color = None

# p.xgrid.visible = False
# p.ygrid.visible = False

output_notebook()
show(p)
# %%
