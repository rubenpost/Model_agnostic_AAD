# %%
import time
import pandas as pd

class encoder:

    def categorical_encoder(preprocessed):
        start = time.time()
        print("Starting categorical encoding..")        
        dummies = pd.get_dummies(preprocessed.object_cols.drop(['EventID', 'case:concept:name', 'OfferID'], axis=1))
        end = time.time()
        print("Categorical encoding took", end - start, "seconds.")
        return pd.concat([dummies, preprocessed.num_cols, pd.DataFrame(preprocessed.case_id_col)], axis=1)

    def numeric_encoder(preprocessed, numeric_encoding):
        start = time.time()
        print("Starting numeric encoding..")
        preprocessed_data = preprocessed.groupby(['case:concept:name'], as_index=False).agg(numeric_encoding)
        preprocessed_data = preprocessed_data.drop(['case:concept:name'], axis=1)
        end = time.time()
        print("Numeric encoding took", end - start, "seconds.")
        return preprocessed_data
# %%
