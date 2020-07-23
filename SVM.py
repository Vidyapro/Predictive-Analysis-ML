# Import Libraries
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Get Data
column_names = ["pregnancies", “glucose",    "bpressure", "skinfold", "insulin", "bmi", "pedigree", "age", "class"]
df = pd.read_csv("data.csv", names=column_names)
print(df.shape)
df.head()

# Extract Features
X = df.iloc[:,:8]
X.head()

# Extract Class Labels
y = df['class']
y.head()

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
X_test.head()

# Normalize Features
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_train[:5,:]

# Training a Support Vector Machine
clf = svm.SVC(kernel='sigmoid')
clf.fit(X_train, y_train)

# Decision Boundary
y_pred = clf.predict(X_train)
print(y_pred)
print(accuracy_score(y_train, y_pred))

# SVM Kernels
for k in ('linear', 'poly', 'rbf', 'sigmoid'):
    clf = svm.SVC(kernel=k)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    print(k)
    print(accuracy_score(y_train, y_pred))

# Instantiating the Best Model
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)

# Making a single prediction
patient = np.array([ [ 1., 100., 75., 40., 0., 45., 1.5, 20. ], ])
patient = scaler.transform(patient)
clf.predict(patient)

# Testing Set Prediction
X_test = scaler.transform(X_test)
patient = np.array([ X_test[1], ])
print(clf.predict(patient))
print(y_test.iloc[1])

#Accuracy on Testing Set
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

#Precision and Recall
print(classification_report(y_test, y_pred))
