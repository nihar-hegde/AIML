import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv('data3.csv')
print('The first 5 values of data is:\n',data.head())

X=data.iloc[:,:-1]
print("\n The first 5 values of the train data set is \n",X.head())

y=data.iloc[:,-1]
print("\n The first 5 values of the train output is \n",y.head())

le_outlook = LabelEncoder()
X.Outlook = le_outlook.fit_transform(X.Outlook)

le_Temperature = LabelEncoder()
X.Temperature = le_Temperature.fit_transform(X.Temperature)

le_Humidity = LabelEncoder()
X.Humidity = le_Humidity.fit_transform(X.Humidity)

le_Wind = LabelEncoder()
X.Wind = le_Wind.fit_transform(X.Wind)

print("\nNow the train output is \n",X.head())

X

le_Answer = LabelEncoder()
y=le_Answer.fit_transform(y)
print("\nNow the train output is \n",y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20)


classifier = GaussianNB()

classifier.fit(X_train,y_train)

from sklearn.metrics import accuracy_score
print("Accuracy is : ",accuracy_score(classifier.predict(X_test),y_test))
