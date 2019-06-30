import numpy as np
import pandas as pd
yeaar=[]
for year in range(2006,2020):
    filename="fsi-{}.xlsx".format(year)
    df= pd.read_excel(filename)
    df= df.drop("Year",axis=1)
    df["year"]=year
    yeaar.append(df)



df1=yeaar[0]

for i in range(1,14):
    df1= pd.concat([df1,yeaar[i]],ignore_index=True)
    
    

#SA= Security apparatus      Eco_ineq= Economic_inequality       HR= Human rights
#FA= Factionalized Elites    HF= Human fight and brain drain     Dp= Demographic pressures
#GG= Group grievence         Sl= State Legitimacy                R_I= Refigee and IDD
#Eco= Economy                PS= Public Services                 EI= External interventions
        
        
df1.columns= ["country","rank","total","SA","FE","GG","Eco","Eco_ineq","HF","SL","PS","HR","DP","R_I","EI","Year"]


features= df1.drop(["rank","total"],axis=1)

features= features.applymap(lambda s : s.lower() if type(s)== str else s)

import re
for i in range(0,2455):
    features["country"][i] = re.sub('[^a-zA-Z]', '', features['country'][i])

'''import pickle
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
features.iloc[:, 0:1] = labelencoder.fit_transform(features.iloc[:,0:1])'''


features1 = features
features=features.values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
features[:,0] = labelencoder.fit_transform(features[:,0])


from sklearn.preprocessing import OneHotEncoder
onehot=OneHotEncoder(categorical_features = [0])
features = onehot.fit_transform(features).toarray()
features=features[:,1:]


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()  
features = sc.fit_transform(features)


features=pd.DataFrame(features)
features_train= features.iloc[:2276,:]
features_test= features.iloc[2276:,:]

labels= df1["total"]
labels=pd.DataFrame(labels)
labels=labels.astype(float)

for i in range(0,2455):
    if( labels["total"][i] < 40.0):
        labels["total"][i]=0
    elif (labels["total"][i] > 60.0):
        labels["total"][i]=2
    else :
        labels["total"][i]=1
        
        
labels_train= labels.iloc[:2276,:]
labels_test= labels.iloc[2276:,:]


from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(features_train, labels_train)



# Predicting the Test set results
labels_pred = classifier.predict(features_test)
classifier.score(features_test, labels_test)


list1=["yemen",10.0,10.0,9.6,9.7,8.1,7.3,9.8,9.8,9.9,9.7,9.6,10.0,2019]
list1=np.array(list1).reshape(1,-1)


import pickle

with open ('model_le', 'wb') as f:
    pickle.dump(labelencoder,f)
with open ('model_le', 'rb') as f1:
    ohe = pickle.load(f1)
    list1[:,0]=ohe.transform(list1[:,0])


with open ('model_ohe', 'wb') as f1:
    pickle.dump(onehot,f1)
with open ('model_ohe', 'rb') as f1:
    ohe = pickle.load(f1)
    list1=ohe.transform(list1).toarray()
    list1=list1[:,1:]


with open ('model_sc', 'wb') as f2:
    pickle.dump(sc,f2)
with open ('model_sc', 'rb') as f2:
    sc1 = pickle.load(f2)


with open ('classifier.pkl', 'wb') as f3:
    pickle.dump(classifier,f3)
with open ('classifier.pkl', 'rb') as f3:
    ohe = pickle.load(f3)

ohe.predict(list1)

"""
from xgboost import XGBClassifier
classifier= XGBClassifier()
classifier.fit(features_train,labels_train)


pred= classifier.predict(features_test)

from sklearn.metrics import confusion_matrix
cm= confusion_matrix(labels_test,pred)

"""


