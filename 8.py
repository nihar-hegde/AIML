import numpy as np #numerical python
import pandas as pd #uused for data analysis
from sklearn.datasets import load_iris #importing iris datasets from datasets module
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

dataset=load_iris()
print(dataset)

x_train,x_test,y_train,y_test = train_test_split(dataset["data"],dataset["target"],random_state=0)
x_train

print(len(dataset["data"]))
print(len(x_train))

kn=KNeighborsClassifier(n_neighbors=1)
kn.fit(x_train,y_train)

kn=KNeighborsClassifier(n_neighbors=1)
kn.fit(x_train,y_train)

for i in range(len(x_test)):
    x=x_test[i]
    x_new=np.array([x])
    prediction = kn.predict(x_new)
    print("TARGET=",y_test[i],dataset["target_names"][y_test[i]],"PREDICTED =",prediction,dataset["target_names"][prediction])
print(kn.score(x_test,y_test))