# %%
import numpy as np

class enrich:

    def bounded_existence(self, activity, count):
        feature_name = 'bounded_existence_%s' % (activity)
        if np.in1d(self['concept:name'], activity).sum() > count: self[feature_name] = int(1)
        else: self[feature_name] = int(0)
        return self

    def binding_duties(self, activity1, activity2):
        feature_name = 'Binding_of_duties_%s_%s' % (activity1, activity2)
        # condition = self.loc[self['concept:name'] == activity1, 'org:resource'][-1:].reset_index(drop=True).eq(self.loc[self['concept:name'] == activity2, 'org:resource'].reset_index(drop=True))
        L1 = self.loc[self['concept:name'] == activity1, 'org:resource'].reset_index(drop=True)
        L2 = self.loc[self['concept:name'] == activity2, 'org:resource'].reset_index(drop=True)
        condition = [i for i in L1.tolist() if i in L2.tolist()]
        # if np.array(condition):    
        if len(condition) >= 1:
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

    def get_average(self, activity):
        test = self['concept:name'].value_counts()
        if activity in test:
            self['average_cancellation'] = test[activity]
        else:
            self['average_cancellation'] = 0
        self['average_resource'] = self['org:resource'].nunique()
        return self
# %%