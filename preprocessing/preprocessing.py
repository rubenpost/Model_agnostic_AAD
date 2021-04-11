# %%
import pandas as pd
import numpy as np
import time

class preprocessor:

    def column_rename(data_path, case_id_col, activity_col, timestamp_col, resource_col):
        start = time.time()

        # load data
        df = pd.read_parquet(data_path)

        # rename mandatory columns
        df.rename(columns={
            case_id_col : 'case:concept:name',
            activity_col : 'concept:name',
            timestamp_col : 'time:timestamp',
            resource_col : 'org:resource'
        }, inplace = True)

        # retype
        df['case:concept:name'] = df['case:concept:name'].astype('str')
        df['concept:name'] = df['concept:name'].astype('str')
        df['time:timestamp'] = df['time:timestamp'].astype(str).str.slice(stop=19)
        df['time:timestamp'] = pd.to_datetime(df["time:timestamp"])
        df['org:resource'] = df['org:resource'].astype('str')

        # Add instance data
        df = datamanager(data = df)

        end = time.time()
        print("Preprocessing took", end - start, "seconds.")
        return df

class datamanager:

    def __init__(self, data):
        # create empty column type lists
        static_cat_cols = []
        dynamic_cat_cols = []
        static_num_cols = []
        dynamic_num_cols = []
        static_cols = []
        dynamic_cols = []

        # Remove 'case:' from column names
        for column_name in data.columns:
            if column_name != 'case:concept:name':
                if column_name[:5] == 'case:':
                    data.rename(columns={column_name:column_name[5:]}, inplace=True)

        # get fist case in the data
        gp = data.groupby('case:concept:name')
        sample = gp.get_group(data['case:concept:name'].iloc[0])
        sample.drop(['time:timestamp'], axis=1, inplace=True)
        sample_object = sample.select_dtypes(include=['object'])
        sample_other = sample.drop(sample_object.columns, axis = 1)

        # base column types based on first case data
        for column in sample_object.columns:
            if sample_object[column].nunique() == 1:
                static_cat_cols.append(str(column))
                static_cols.append(str(column))
            else:
                dynamic_cat_cols.append(str(column))
                dynamic_cols.append(str(column))

        for column in sample_other.columns:
            if sample_other[column].nunique() == 1:
                static_num_cols.append(str(column))
                static_cols.append(str(column))
            else:
                dynamic_num_cols.append(str(column))
                dynamic_cols.append(str(column))
        
        # Remove boolean columns from numeric and append to categoric

        # base attributes based column types
        self.static_cat_cols = data[static_cat_cols]
        self.dynamic_cat_cols = data[dynamic_cat_cols]
        self.static_num_cols = data[static_num_cols]
        self.dynamic_num_cols = data[dynamic_num_cols]
        self.static_cols = data[static_cols]
        self.dynamic_cols = data[dynamic_cols]
        self.num_cols = data.select_dtypes(include='number')
        self.object_cols = data.select_dtypes(include='object')
        self.case_id_col = data['case:concept:name']
        self.data = data

# %%
 