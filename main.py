
# %%
# Final version
import numpy as np
import pandas as pd
from random import randint
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
preprocessed.data['concept:name'].replace({'SELECTED':'Loan requested',
                                            'CREATED':'Create loan offer',
                                            'SENT':'Sent loan offer to client',
                                            'SENT_BACK':'Receive documents from client',
                                            'ACCEPTED':'Approve loan request',
                                            'CANCELLED':'Cancel loan request'}, inplace=True)

preprocessed = datamanager(data = preprocessed.data)
preprocessed.num_cols.drop(['Unnamed: 0'], axis=1, inplace=True)
preprocessed.data.drop(['Unnamed: 0'], axis=1, inplace=True)

# Encode data
numeric_encoding = {
    'Request loan amount':'last',
    'Case length in calendar days':'max',
    'activity_count':'max',
    'average_cancellation':'max',
    'average_resource':'max',
    'activity_count':'max',
    'average_cancellation':'max',
    'average_resource':'max',
    'bounded_existence_W_Beoordelen fraude':'max',
    'Binding_of_duties_O_CREATED_O_ACCEPTED':'max'}

encoded_numeric = encoder.numeric_encoder(preprocessed, numeric_encoding)
encoded_categorical = encoder.categorical_encoder(preprocessed)
encoded_data = pd.concat([encoded_numeric, encoded_categorical.reset_index()], axis=1).fillna(0)

# Detect anomalies 
aad = detect_anomalies(np.asarray(encoded_data), preprocessed)

# %%
