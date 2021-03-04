# %%
import numpy as np

class filter:

    def bounded_existence(self, activity):
        feature_name = 'bounded_existence_%s' % (activity)
        if np.in1d(self['concept:name'], activity).sum() > 3: self[feature_name] = 1
        else: self[feature_name] = 0
        return self

    def four_eye_principle(self, activity1, activity2):
        feature_name = 'bounded_existence_%s_%s' % (activity1, activity2)
        if (len(self[self['concept:name'].isin([activity1])])) and (len(self[self['concept:name'].isin([activity2])])) == 1:
            if self.loc[self['concept:name'] == activity1, 'org:resource'].iloc[0] != self.loc[self['concept:name'] == activity2, 'org:resource'].iloc[0]:
                self[feature_name] = 1
        else:
            self[feature_name] = 0
        return self 

    def access_control(self, activity, resource):
        feature_name = 'bounded_existence_%s_%s' % (activity, resource)
        if len(self[self['concept:name'].isin([activity])]) >= 1:
            if self.loc[self['concept:name'] == activity, 'org:resource'].iloc[0] != resource:
                self[feature_name] = 1
        else:
            self[feature_name] = 0
        return self   
# %%