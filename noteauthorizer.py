# Importing the dependencies

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle


# Loading the dataset into a pandas dataframe
bank_note = pd.read_csv('Data.csv')
# print(bank_note.head(3))
# print(bank_note.shape)
# print(bank_note.describe())
# print(bank_note.isnull().sum())

X = bank_note.iloc[:, :-1]  # Independent variables
Y = bank_note.iloc[:, -1]  # Dependent variables

# print(X.head)
# print(Y.head)
# print(Y.unique())

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=2)
#print(X_train.shape, X_test.shape, X.shape)


# Training the model
classifier = RandomForestClassifier()
classifier.fit(X_train, Y_train)

X_train_predict = classifier.predict(X_train)
Train_test_score = accuracy_score(Y_train, X_train_predict)
# print(Train_test_score*100)

# Testing data prediction
X_test_predict = classifier.predict(X_test)
Test_data_score = accuracy_score(Y_test, X_test_predict)
print('Testing Data Accuracy:', Test_data_score*100)

input_data = (-2.0149,3.6874,-1.9385,-3.8918)

new_array = np.array(input_data).reshape(1, -1)
prediction = classifier.predict(new_array)
print(prediction)

if prediction [0]== 1:
 print('Note is authentic')
else:
 print('Note is not authentic')


# Loading the trained model into a pickle file
filename = 'banknote_model.pkl'
pickle.dump(classifier, open('banknote_model.pkl', 'wb'))

loaded_model = pickle.load(open('banknote_model.pkl', 'rb'))

input_data = (-2.0149, 3.6874, -1.9385, -3.8918)

new_array = np.array(input_data).reshape(1, -1)
prediction = classifier.predict(new_array)
print(prediction)

if prediction[0] == 1:
    print('Note is authentic')
else:
    print('Note is not authentic')
