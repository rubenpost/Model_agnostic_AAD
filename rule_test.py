# %%
import sys
sys.path.insert(1,"/workspaces/thesis/compliance_analysis/compliance_rules")
from compliance_analysis.compliance_rules.compliance_checker import Filter
import pandas as pd
df = pd.read_parquet('/workspaces/thesis/data/full_2012.gzip')

# %%
df = df.groupby(['case:concept:name']).apply(Filter.bounded_existence, activity = 'W_Completeren aanvraag')
#df = df.groupby(['case:concept:name']).apply(Filter.four_eye_principle, activity1 = 'A_SUBMITTED', activity2 = 'A_PREACCEPTED')
#df = df.groupby(['case:concept:name']).apply(Filter.access_control, activity = 'W_Beoordelen fraude')
# %%
df['four_eye_principle'].value_counts().sum()
# %%
len(df[df['concept:name'].str.contains('A_')]['concept:name'].unique())
# %%
