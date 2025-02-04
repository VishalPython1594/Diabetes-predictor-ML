import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

dataset_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

df = pd.read_csv(dataset_url, names = column_names)

x = df.iloc[:,:-1]
y = df.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 101)

model = LogisticRegression()
model.fit(x_train, y_train)
print('[INFO] model trained')

#accuracy
acc = model.score(x_test, y_test)
print(f'[INFO] model accuracy : {acc}')


#joblib
import joblib

joblib.dump(model, 'dib_model.pkl')
print('[INFO] model saved')