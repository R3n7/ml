import sklearn

import sklearn
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model,preprocessing

data = pd.read_csv('car.data',sep=',')
#print(data.head())
'''
In this dataset the attributes are all string but we need them 
as numbers for computation.so we can either change them into numbers 
correspondingly by using preprocessing function from sklearn
'''
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data['buying']))
maint = le.fit_transform(list(data['maint']))
doors = le.fit_transform(list(data['doors']))
persons = le.fit_transform(list(data['persons']))
lug_boot = le.fit_transform(list(data['lug_boot']))
safety = le.fit_transform(list(data['safety']))
clss = le.fit_transform(list(data['class']))
print(buying)

predict = "clss"
x = list(zip(buying,maint,doors,persons,lug_boot,safety))
y = list(clss)
x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)

model = KNeighborsClassifier(n_neighbors=7)

model.fit(x_train,y_train)
acc = model.score(x_test,y_test)
print(acc)
predicted = model.predict(x_test)
names =["unacc","acc","good","vgood"]
for x in range(len(x_test)):
    print("Predicted: ",names[predicted[x]], "Data: ",x_test[x],"Actual: ",names[y_test[x]])
    n = model.kneighbors([x_test[x]],7,True)#to show the nrighbours of the points
    print("N: ",n)