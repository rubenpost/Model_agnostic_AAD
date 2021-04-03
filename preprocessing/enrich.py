# %%
import numpy as np

class enrich:

    def bounded_existence(self, activity):
        feature_name = 'bounded_existence_%s' % (activity)
        if np.in1d(self['concept:name'], activity).sum() > 3: self[feature_name] = int(1)
        else: self[feature_name] = int(0)
        return self

    def four_eye_principle(self, activity1, activity2):
        feature_name = 'four_eye_principle_%s_%s' % (activity1, activity2)
        condition = self.loc[self['concept:name'] == activity1, 'org:resource'][-1:].reset_index(drop=True).eq(self.loc[self['concept:name'] == activity2, 'org:resource'].reset_index(drop=True))
        if np.array(condition):    
            self[feature_name] = int(1)
        else:
            self[feature_name] = int(0)
        return self

    def access_control(self, activity, resource):
        feature_name = 'access_control_%s_%s' % (activity, resource)
        if len(self[self['concept:name'].isin([activity])]) >= 1:
            if self.loc[self['concept:name'] == activity, 'org:resource'].iloc[0] != resource:
                self[feature_name] = int(1)
        else:
            self[feature_name] = int(0)
        return self   
    
    def feature_engineering(self):
        case_length = self['time:timestamp'].max() - self['time:timestamp'].min()
        self['Case length in calendar days'] = ((case_length.total_seconds()/60)/60)//24
        self['Case length in calendar days'] = round(self['Case length in calendar days'])
        self['activity_count'] = len(self['concept:name'])
        return self
# %%