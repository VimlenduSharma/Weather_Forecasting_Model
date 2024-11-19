#!/usr/bin/env python
# coding: utf-8

# In[105]:


get_ipython().system('pip install numpy')
get_ipython().system('pip install pandas')


# In[106]:


import pandas as pd #data preprocessing,csv file i/0
import numpy as np#linear algebra


# In[107]:


df=pd.read_csv('Weather Training Data.csv')


# In[108]:


print('Size of weather data frame is:',df.shape)


# In[109]:


print(df[0:5])


# In[110]:


#checking null values
#data_preprocessing
print(df.count().sort_values())


# In[111]:


#removing_unwanted_variables
df=df.drop(columns=['Sunshine','Evaporation','Cloud3pm','Cloud9am','Location'],axis=1)
print(df.shape)


# In[112]:


#get_rid_of_null_values
df=df.dropna(how='any')
print(df.shape)


# In[113]:


#remove_outliers_in_data_using_z_score
from scipy import stats
z=np.abs(stats.zscore(df._get_numeric_data()))
print(z)
df=df[(z<3).all(axis=1)]
print(df.shape)


# In[114]:


#for-categorical_columns_change_yes_no_to_1/0_for_rain_today_and_rain_tommorrow
df['RainToday'].replace({'No':0,'Yes':1},inplace=True)
df['RainTomorrow'].replace({'No':0,'Yes':1},inplace=True)
print(df.shape)
print(df)


# In[115]:


#change_unique_values_to_int
categorical_columns=['WindDir3pm']
for col in categorical_columns:
    print(np.unique(df[col]))
    #transform_categorical_columns
df=pd.get_dummies(df,columns=categorical_columns)
print(df.iloc[4:9])


# In[116]:


#standardise_data_using_min_max_scaler
# Identify non-numeric columns
# Import the MinMaxScaler
from sklearn import preprocessing
# Identify and convert categorical columns to numeric using label encoding
categorical_columns = df.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_columns:
    le = preprocessing.LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

scaler = preprocessing.MinMaxScaler()
scaler.fit(df)
df = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)

print(df.iloc[4:10])


# In[117]:


#Preprocessing_is_complete
#Expolatory_data_analysis
#feature_selection
#selectKBest_function is used to select some selective variables
from sklearn.feature_selection import SelectKBest ,chi2


# In[16]:


#it will select the most significant predictor variable
x=df.loc[:,df.columns!='RainTomorrow']
y=df[['RainTomorrow']]
selector=SelectKBest(chi2,k=3)
selector.fit(x,y)
x_new=selector.transform(x)
print(x.columns[selector.get_support(indices=True)])#top 3 columns


# In[17]:


#get_hold_of_important_features_and_assign_them_as_x
df=df[['Humidity3pm','Rainfall','RainToday','RainTomorrow']]
x=df[['Humidity3pm']]#let's_use_only_one_feature
y=df[['RainTomorrow']]


# In[18]:


#data_modelling
#use_classification_logisitic_regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time


# In[19]:


#calculate_accuracy_and_time_taken_by_classifier
t0=time.time()


# In[20]:


#data_splicing-splitting_data_in_testing_and_training_data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)#testing-data=25%and_remaining_training_data=75%
clf_logreg=LogisticRegression(random_state=0)#creation_of_instance_for_logistic_regression
#fit/build the model_using_training_dataset
clf_logreg.fit(x_train,y_train)


# In[21]:


#evaluating_model_using_testing_dataset
y_pred=clf_logreg.predict(x_test)
score1=accuracy_score(y_test,y_pred)


# In[22]:


print('accuracy using logistic regression:',score1)
print('time taken by logistic regression:',time.time()-t0)


# In[23]:


#random_forest_classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# In[24]:


#calculating_accuarcy_and_time_taken_by_classifier
t0=time.time()


# In[25]:


#data_splitting
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
clf_rf=RandomForestClassifier(n_estimators=100,max_depth=4,random_state=0)
clf_rf.fit(x_train,y_train)#fitting the model
y_pred=clf_rf.predict(x_test)
score2=accuracy_score(y_test,y_pred)
print('Accuracy by Random Forest Classifier:',score2)
print('Time Taken by Random Forest Classifier:',time.time()-t0)


# In[26]:


#Decision_tree_classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# In[27]:


t0=time.time()


# In[28]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
clf_dt=DecisionTreeClassifier(random_state=0)
clf_dt.fit(x_train,y_train)
#evaluate_model_using_testing_data
y_pred=clf_dt.predict(x_test)
score3=accuracy_score(y_test,y_pred)


# In[29]:


print('Accuracy by Decision Tree Classifier:',score3)
print('Time Taken by classifier:',time.time()-t0)


# In[30]:


#Support_vector_machine
from sklearn import svm
from sklearn.model_selection import train_test_split


# In[31]:


t0=time.time()


# In[32]:


#data_splitting
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
clf_svc=svm.SVC(kernel='linear')


# In[33]:


#builiding/fitting_model_using_training_data
clf_svc.fit(x_train,y_train)
#evaluating_the_model
y_pred=clf_svc.predict(x_test)
score4=accuracy_score(y_test,y_pred)


# In[34]:


print('Accuracy By Support Vector Machine:',score4)


# In[35]:


print('Accuracy By Support Vector Machine:',score4)
print('Time taken By Support Vector Machine:',time.time()-t0)


# In[36]:


get_ipython().system('pip install tensorflow')
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense


# In[37]:


model = Sequential()


# In[38]:


import numpy as np
import matplotlib.pyplot as plt


# In[39]:


score = np.float64(0.85)

# Check the type of the variable
if isinstance(score, dict):
    names = list(score.keys())
    values = list(score.values())
    
    plt.figure(figsize=(10, 5))
    plt.bar(names, values)
    plt.xlabel('Classifiers')
    plt.ylabel('Accuracy Score')
    plt.title('Accuracy Score of Different Classifiers')
    plt.ylim(0, 1.0)  # Set y-axis limit to 0-1 for accuracy scores
    plt.show()
else:
    print("The variable is not a dictionary.")


# In[40]:


# Deep Learning Model (LSTM)
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(1, 1)))  # Adjust input shape based on your data
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# In[41]:


x_train_lstm = np.reshape(x_train.values, (x_train.shape[0], 1, 1))
x_test_lstm = np.reshape(x_test.values, (x_test.shape[0], 1, 1))


# In[42]:


t0 = time.time()
model.fit(x_train_lstm, y_train, epochs=50, verbose=1)
predictions = model.predict(x_test_lstm)
score_lstm = accuracy_score(y_test, predictions.round())
print(f'Accuracy using LSTM: {score_lstm}')
print(f'Time taken by LSTM: {time.time() - t0}')


# In[43]:


import matplotlib.pyplot as plt


# In[44]:


#artificial_neural_network
model = Sequential()
model.add(Dense(32, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
y_pred = (model.predict(x_test) > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy on the test set: {accuracy}")


# In[48]:


data=pd.DataFrame({'Classifier_Name':{1:'Logistic_Regression',2:'Random_forest',3:'Decision_Tree',4:'Support_Vector_Machine',5:'LSTM',6:'Artificial_Neural_Network'},'Accuracy':{1:[score1],2:[score2],3:[score3],4:[score4],5:[score_lstm],6:[accuracy]}})
data


# In[49]:


#for_lstm
history = model.fit(x_train_lstm, y_train, epochs=50, verbose=1)

# Plot the training loss over epochs
plt.plot(history.history['loss'])
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Evaluate the model on the test set
predictions = model.predict(x_test_lstm)
score_lstm = accuracy_score(y_test, predictions.round())
print(f'Accuracy using LSTM: {score_lstm}')


# In[50]:


#for_ANN_plot
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score

# Assuming x_train, y_train, x_test, and y_test are your training and testing data

# Define the model
model = Sequential()
model.add(Dense(32, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model and record the training history
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Plot the training and validation accuracy over epochs
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()

# Evaluate the model on the test set
y_pred = (model.predict(x_test) > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy on the test set: {accuracy}")


# In[52]:


#k_n_n_classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
k = 3  # Choose the value of k
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(x_train,y_train)


# In[53]:


y_pred = knn_classifier.predict(x_test)


# In[56]:


accuracy1 = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy1:.2f}')

print('\nConfusion Matrix:')
print(confusion_matrix(y_test, y_pred))

print('\nClassification Report:')
print(classification_report(y_test, y_pred))


# In[57]:


data=pd.DataFrame({'Classifier_Name':{1:'Logistic_Regression',2:'Random_forest',3:'Decision_Tree',4:'Support_Vector_Machine',5:'LSTM',6:'Artificial_Neural_Network',7:'K_N_N_Classifier'},'Accuracy':{1:[score1],2:[score2],3:[score3],4:[score4],5:[score_lstm],6:[accuracy],7:[accuracy1]}})
data


# In[58]:


d1 = pd.DataFrame({
    'Classifier_Name': {
        1: 'Logistic_Regression',
        2: 'Random_forest',
        3: 'Decision_Tree',
        4: 'Support_Vector_Machine',
        5: 'LSTM',
        6: 'Artificial_Neural_Network',
        7: 'K_N_N_Classifier'
    },
    'Accuracy': {
        1: [score1],
        2: [score2],
        3: [score3],
        4: [score4],
        5: [score_lstm],
        6: [accuracy],
        7: [accuracy1]
    }
})

# maximum accuracy and its corresponding classifier
max_accuracy_row = data.loc[data['Accuracy'].apply(lambda x: max(x)).idxmax()]

# Print the results
print(f"The maximum accuracy is {max_accuracy_row['Accuracy'][0]} achieved by {max_accuracy_row['Classifier_Name']}.")


# In[59]:


plt.figure(figsize=(10, 6))
plt.bar(data['Classifier_Name'], data['Accuracy'].apply(lambda x: max(x)))
plt.xlabel('Classifier Name')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison Across Classifiers')
plt.ylim(0, 1)  # Set the y-axis limit to better visualize differences
plt.show()


# In[69]:


pip install seaborn


# In[67]:


pip install pandas


# In[68]:


print("Machine Learning Project has been Finished!")


# In[79]:


print("End!")


# In[80]:


print("Best_Accuracy!")


# In[83]:


df1=pd.read_csv('Weather Test Data.csv')


# In[84]:


df1


# In[86]:


df1=df1.drop(columns=['Sunshine','Evaporation','Cloud3pm','Cloud9am','Location'],axis=1)
print(df1.shape)


# In[87]:


df1=df1.dropna(how='any')
print(df1.shape)


# In[88]:


from scipy import stats
z=np.abs(stats.zscore(df1._get_numeric_data()))
print(z)
df1=df1[(z<3).all(axis=1)]
print(df1.shape)


# In[90]:


df1['RainToday'].replace({'No':0,'Yes':1},inplace=True)
print(df1.shape)
print(df1)


# In[94]:


categorical_columns=['WindDir9am']
for col in categorical_columns:
    print(np.unique(df1[col]))
    #transform_categorical_columns
df1=pd.get_dummies(df1,columns=categorical_columns)
print(df1.iloc[4:9])


# In[95]:


from sklearn import preprocessing
# Identify and convert categorical columns to numeric using label encoding
categorical_columns = df1.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_columns:
    le = preprocessing.LabelEncoder()
    df1[col] = le.fit_transform(df1[col])
    label_encoders[col] = le

scaler = preprocessing.MinMaxScaler()
scaler.fit(df1)
df1 = pd.DataFrame(scaler.transform(df1), index=df1.index, columns=df1.columns)

print(df1.iloc[4:10])


# In[98]:


df1=df1[['Humidity3pm','Rainfall','RainToday']]
x1=df1[['Humidity3pm']]#let's_use_only_one_feature
y1=df1[['RainToday']]


# In[99]:


x_train,x_test,y_train,y_test=train_test_split(x1,y1,test_size=0.25)
clf_svc=svm.SVC(kernel='linear')


# In[100]:


clf_svc.fit(x_train,y_train)
#evaluating_the_model
y_pred=clf_svc.predict(x_test)
score4=accuracy_score(y_test,y_pred)


# In[101]:


predictions = clf_svc.predict(x_test)


# In[102]:


print("prediction_value",predictions)


# In[1]:





# In[ ]:




