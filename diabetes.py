from flask import Flask, render_template, request
import joblib

model = joblib.load('dib_model.pkl')

#init(initialize the app)
app= Flask(__name__)

@app.route('/')
def home():
    return render_template('dia.html')

@app.route('/submit', methods = ['post'])
def submit():
    data = [float(data) for data in request.form.values()]
    print(data)
    res = model.predict([data])
    print(f'result is --> {res}')
    
    if res[0] == 1:
        return 'Oops! Found an increase in the sugar level! Take Care!'
    
    return 'Hooray! you deserve at least one gulab jamun today!'

app.run(debug=True)
