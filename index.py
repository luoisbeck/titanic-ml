
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from random import randint
import re



hist_data = pd.read_csv('train.csv', sep=',')
hist_data_test = pd.read_csv('test.csv', sep=',')

print(hist_data.size)
print(hist_data.head(5))


X_data = hist_data.drop(hist_data.columns[[1]], axis=1)
new_X_data = pd.concat([X_data, hist_data_test], axis=0)
print(new_X_data)



passenger_titles = hist_data['Name'].values
# names = names.reshape(-1,1)
print(type(passenger_titles))
regex = r', (.*)\. '
for name_index in range(0,len(passenger_titles)):
    
    # print(names[name_index])
    passenger_titles[name_index] = re.search(regex, passenger_titles[name_index]).group(1)
    
passenger_titles = passenger_titles.reshape(-1,1)
print(passenger_titles[0:10,:])



X_data = hist_data.drop(hist_data.columns[[0,3,8,10,11]], axis=1)
Y_data = hist_data['Survived'].values
print(X_data.shape)
Y_data = Y_data.reshape(-1,1)
print(Y_data.shape)
print(X_data.head(2))
print(X_data)


# In[15]:


passenger_titles = pd.DataFrame(passenger_titles, columns=['PassengerTitles'])
# X_data = pd.DataFrame(X_data)
X_data = pd.concat([passenger_titles, X_data], axis=1)
# print(X_data)

le = LabelEncoder()
enc = OneHotEncoder(sparse=False)

for col in X_data.columns.values:
    if X_data[col].dtypes=='object':
        le.fit(X_data[col].values)
        X_data[col] = le.transform(X_data[col])

ohe_list = ['PassengerTitles','Pclass','Sex']
# print(X_data.iloc[:, ohe_list])
# print(X_data)
for col in ohe_list:
    enc.fit(X_data[[col]])
    temp = enc.transform(X_data[[col]])
    
    temp = pd.DataFrame(temp, columns=[(col+"_"+str(i)) for i in X_data[col].value_counts().index])
    X_data = pd.concat([X_data, temp], axis=1)

temp.head()
X_data.head()


# In[16]:


X_data = X_data.values


# We have to use this in order to deal with NaN values
my_imputer = Imputer()
X_data = my_imputer.fit_transform(X_data)
# print(X_data)


# In[17]:


rand_state_seed = randint(0,100)
# scaler = MinMaxScaler()
# X_data = scaler.fit_transform(X_data)
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.7, random_state=rand_state_seed)


# In[18]:

C = [0.01,0.1,1.,10.,20.]

print("SVM")
kernel_type = ['rbf', 'linear', 'sigmoid']
for c_value in C:
    for kern in kernel_type:
        clf = SVC(kernel=kern, C=c_value)  # , degree=int(degree_value))
        clf.fit(X_train, Y_train)
        print("SVM    !!!    State: " + str(rand_state_seed) + 'Kernel: ' + kern + " - C: " + str(c_value) + " - Score: " + str(clf.score(X_test, Y_test)))
        # print("Prediction: " + str(clf.predict(predict_me)))


print("Logistical Regression")
for c_value in C:
    clf = LogisticRegression(C=c_value)  # , degree=int(degree_value))
    clf.fit(X_train, Y_train)
    print("Logistical Regression   !!!   State: " + str(rand_state_seed) + " - C: " + str(c_value) + " - Score: " + str(clf.score(X_test, Y_test)))

"""
print("Decision Tree")
clf = DecisionTreeClassifier()  # , degree=int(degree_value))
clf.fit(X_train, Y_train)
print("State: " + str(rand_state_seed) + " - Score: " + str(clf.score(X_test, Y_test)))
"""

print("Neural Network MLPC Classifier")
solver_mlpc = ['lbfgs', 'sgd', 'adam']
activation_list = ['identity', 'logistic', 'tanh', 'relu']

for solver in solver_mlpc:
    for activation_key in activation_list:
        clf = MLPClassifier(solver=solver, activation=activation_key,alpha=1e-5, hidden_layer_sizes=(200,100,200,10), random_state=1)
        clf.fit(X_train, Y_train)
        print("Neural Networks   !!!   State: " + str(rand_state_seed) + 'Solver :'+ solver + ' - Activation: ' + activation_key + " - Score: " + str(clf.score(X_test, Y_test)))

