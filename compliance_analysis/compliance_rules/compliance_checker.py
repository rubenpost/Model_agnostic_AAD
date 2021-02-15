# %%
import numpy as np

class Filter:
    
    def __init__(self, name): #in case I ever wanna attach attributes to self
        self.name = name

    def bounded_existence(self, activity):
        if np.in1d(self['concept:name'], activity).sum() > 3: self['bounded_existence'] = 1
        else: self['bounded_existence'] = 0
        return self

    def four_eye_principle(self, activity1, activity2):
        if (len(self[self['concept:name'].isin([activity1])])) and (len(self[self['concept:name'].isin([activity2])])) == 1:
            if self.loc[self['concept:name'] == activity1, 'org:resource'].iloc[0] != self.loc[self['concept:name'] == activity2, 'org:resource'].iloc[0]:
                self['four_eye_principle'] = 1
        else:
            self['four_eye_principle'] = 0
        return self 

    def access_control(self, activity):
        if len(self[self['concept:name'].isin([activity])]) >= 1:
            if self.loc[self['concept:name'] == activity, 'org:resource'].iloc[0] != 112:
                self['access_control'] = 1
        else:
            self['access_control'] = 0
        return self   
# %%