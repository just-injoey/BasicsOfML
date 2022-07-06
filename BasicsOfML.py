# BasicsOfML
# from statistics import linear_regression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import *

data = pd.read_csv(r"C:\Users\HELLO\OneDrive\Desktop\coding\LinkedinDataPredictionCourse\insurance.csv")
print(data.head(15))

sex = data.iloc[:,1:2].values
smoker = data.iloc[:,4:5].values



le = LabelEncoder()
sex[:,0] = le.fit_transform(sex[:,0])
sex = pd.DataFrame(sex)
sex.columns = ['sex']
le_sex_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Sklearn label encoding for sex")
print(le_sex_mapping)
print(sex[:10])

smoker[:,0] = le.fit_transform(smoker[:,0])
smoker = pd.DataFrame(smoker)
smoker.columns = ['smoker']
le_smoker_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
# print("Sklearn label encoding for smoker")
# print(le_smoker_mapping)
# print(smoker[:10])

region = data.iloc[:,5:6].values
ohe = OneHotEncoder()
region = ohe.fit_transform(region).toarray()
region = pd.DataFrame(region)
region.columns = ['northeast', 'northwest', 'southeast', 'sothwest']
print("Sklearn one hot encoding for region")
print(region[:10])

X_num = data[['age','bmi','children']]
X_final = pd.concat([X_num, sex, smoker, region], axis=1)

y_final = data[['charges']].copy()

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size= 0.33, random_state=0)


# # normalisation scale
# n_scaler = MinMaxScaler()
# X_train = n_scaler.fit_transform(X_train.astype(np.float64))
# X_test = n_scaler.transform(X_test.astype(np.float64))

#standardisation scale
s_scaler = StandardScaler()
X_train = s_scaler.fit_transform(X_train.astype(np.float64))
X_test = s_scaler.transform(X_test.astype(np.float64))

# print(X_train)
print(pd.DataFrame(X_test))         


# linear regression
# LinearRegression
# lr = linear_regression().fit(X_train, y_train)
lr = LinearRegression().fit(X_train, y_train)
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

print(f"lr.coef_:{lr.coef_}")
print(f"lr.intercept_: {lr.intercept_}")
print('lr train score %.3f, lr test score %.3f' % (lr.score(X_train,y_train), lr.score(X_test,y_test)))