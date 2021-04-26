# %%
import pandas as pd
# %%
df = pd.read_csv('/workspaces/thesis/AAD Evaluation_April 23, 2021_02.54.csv')
numbers = ('1', '2', '3', '4', '5', '6', '7', '8', '9')
select_col = [col for col in df if col.startswith(numbers)]
df = df[select_col].loc[2:]
# %%
df
# %%
