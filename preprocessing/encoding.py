# %%
import time
import pandas as pd

class encoder:

    def categorical_encoder(self):
        start = time.time()
        print("Starting categorical encoding..")      
        encoded_categorical = self.object_cols.groupby(['case:concept:name']).progress_apply(encoder.categorical_encoder_apply, preprocessed = self)
        end = time.time()
        print("Categorical encoding took", end - start, "seconds.")
        return encoded_categorical

    def categorical_encoder_apply(self, preprocessed):
        encoded_categorical = pd.DataFrame()
        for column in preprocessed.dynamic_cat_cols.columns:#drop(['EventID'], axis=1).columns:
            dynamic_object = self[column].value_counts().reset_index()
            dynamic_object = dynamic_object.pivot_table(columns='index', values=column).reset_index()
            dynamic_object.drop('index', axis=1, inplace=True)
            encoded_categorical = pd.concat([encoded_categorical, dynamic_object], axis=1)
        encoded_categorical = encoded_categorical.reset_index()
        return encoded_categorical

    def numeric_encoder(self, numeric_encoding):
        start = time.time()
        print("Starting numeric encoding..")
        preprocessed_data = self.data.groupby(['case:concept:name'], as_index=False).agg(numeric_encoding)
        preprocessed_data = preprocessed_data.drop(['case:concept:name'], axis=1)
        end = time.time()
        print("Numeric encoding took", end - start, "seconds.")
        return preprocessed_data
# %%
