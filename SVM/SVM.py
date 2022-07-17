#support vector machine
import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
cancer = datasets.load_breast_cancer()

#print(cancer.feature_names)
#print(cancer.target)

x = cancer.data
y = cancer.target

x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.2)

#print(x_train,y_train)
classes = ['malignant','benign']

clf = svm.SVC(kernel='linear')
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
acc = metrics.accuracy_score(y_test,y_pred)
print(acc)