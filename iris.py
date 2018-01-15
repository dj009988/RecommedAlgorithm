from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
filename="iris.data.csv"
names=['sepal-length','sepal-width','petal-length','petal-width','class']
dataset=read_csv(filename,names=names)
print("数据维度:行(%d) 列(%d)"%dataset.shape)
array=dataset.values
X=array[:,0:4]
Y=array[:,4]
validation_size=0.2
seed=7
X_train,X_validation,Y_train,Y_validation=\
train_test_split(X,Y,test_size=validation_size,random_state=seed)
models={}
models['LR']=LogisticRegression()
models['LDA']=LinearDiscriminantAnalysis()
models['KNN']=KNeighborsClassifier()
models['CART']=DecisionTreeClassifier()
models['NB']=GaussianNB()
models['SVM']=SVC()
results=[]
for key in models:
    kfold=KFold(n_splits=10,random_state=seed)
    cv_results=cross_val_score(models[key],X_train,Y_train,
                               cv=kfold,scoring='accuracy')
    results.append(cv_results)
    print("%s:%f(%f)"%(key,cv_results.mean(),cv_results.std()))
    
