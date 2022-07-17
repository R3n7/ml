import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import  style
data = pd.read_csv('student-mat.csv',sep=';')
#print(data.head())
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
predict='G3'
x=np.array(data.drop([predict],1))
y =np.array(data[predict])
x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)
'''
#to get a model that is trained with high accuracy
best =0
for _ in range(30):
    x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train,y_train)
    acc = linear.score(x_test,y_test)
    print(acc)
    if acc>best:
        best = acc
        #we save the models we trained
        with open('studmodel1.pickle','wb') as f:
            pickle.dump(linear,f)
'''
pickle_in = open('studmodel1.pickle','rb') #to load model from pickle files
linear = pickle.load(pickle_in) # to use them
print("co: ",linear.coef_)
print("intercept: ",linear.intercept_)
predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x],x_test[x],y_test[x])

#to show them in a another manner(graphically using matplotlib)
p = 'G1'
style.use('ggplot')
pyplot.scatter(data[p],data["G3"])
pyplot.xlabel(p)
pyplot.ylabel('Final Grade')
pyplot.show()
#we can use this for any graphplotting.
'''
we can see the relationship among the parameters.
how they affect the final grade(which we are trying to predict) 
'''