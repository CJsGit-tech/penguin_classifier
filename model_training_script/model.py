import pandas as pd 
import numpy as np 

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('../data/penguins.csv')
df.dropna(inplace = True)

X = df[['island', 'bill_length_mm', 'bill_depth_mm','flipper_length_mm', 'body_mass_g', 'sex']]
# Auto-Detect Categorical variables and make it into onehot encoding
features = pd.get_dummies(X)

# print(features)

y = df['species']
# Autodetect Categorical Variables and make it into numerical class code (0,1,2,3....)
target, classes = pd.factorize(y)

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(features,target,test_size=0.2,random_state=101,stratify=target)
clf = RandomForestClassifier(random_state=101)
clf.fit(x_train.values,y_train)
acc_score = clf.score(x_test.values,y_test)
print('Our accuracy score for this model is {}'.format(acc_score))

# Serialize Data into Binary Format for later import by using pickle
"""
Pickle in Python is primarily used in serializing and deserializing a Python object structure. 
In other words, it’s the process of converting a Python object into a byte stream to store it in a file/database, 
maintain program state across sessions, or transport data over the network.
"""
import pickle 

rf_pickle = open('random_forest/random_forest_model.pickle', 'wb')
pickle.dump(clf, rf_pickle)
rf_pickle.close()

output_pickle = open('random_forest/classes_map.pickle', 'wb')
pickle.dump(classes, output_pickle)
output_pickle.close() 