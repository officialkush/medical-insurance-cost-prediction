from flask import Flask,request,render_template

import pickle
import os
import numpy as np 


app= Flask(__name__)


model=pickle.load(open('savedmodel.sav','rb'))
ct=pickle.load(open('encoder.sav','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    age=int(request.form["age"])
    sex=request.form['sex']
    bmi=float(request.form['bmi'])
    children=int(request.form['children'])
    smoker=request.form['smoker']
    region=request.form['region']

    data=[[
        age,
        sex,
        bmi,
        children,
        smoker,
        region

    ]]

    final_input=ct.transform(data)
    final_predict=model.predict(final_input)
    result = round(final_predict[0], 2)

    

    return render_template(
        "index.html",
        prediction_text=f"prediction is {result}"
    )
        

if __name__=='__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)