import joblib

#load the model
model = joblib.load('dib_model.pkl')
print('[INFO] model loaded')

#prediction
result = model.predict([[1,1,1,1,1,1,1,1]])
print(result)
