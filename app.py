import pickle
import jsonify
import requests
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split




data=pd.read_csv('voice_recognition.csv')

X=data.drop(['label'],axis=1)
y=data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10)

gboost_model = GradientBoostingClassifier(n_estimators = 100, max_depth = 4, learning_rate= 0.9, random_state = 10)
gboost_model.fit(X_train, y_train)


filename = 'finalized_model.pkl'
pickle.dump(gboost_model, open(filename, 'wb'))


app = Flask(__name__)
model = pickle.load(open('finalized_model.pkl', 'rb'))


@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    
    if request.method == 'POST':
        
        meanfreq=float(request.form['meanfreq'])
        sd=float(request.form['sd'])
        median=float(request.form['median'])
        Q25=float(request.form['Q25'])
        Q75=float(request.form['Q75'])
        skew=float(request.form['skew'])
        sp_ent=float(request.form['sp.ent'])
        sfm=float(request.form['sfm'])
        mode=float(request.form['mode'])
        meanfun=float(request.form['meanfun'])
        minfun=float(request.form['minfun'])
        maxfun=float(request.form['maxfun'])
        meandom=float(request.form['meandom'])
        mindom=float(request.form['mindom'])
        dfrange=float(request.form['dfrange'])
        modindx=float(request.form['modindx'])
        
        

        def lr(meanfreq,sd,median,Q25,Q75,skew,sp_ent,sfm,mode,meanfun,minfun,maxfun,meandom,mindom,dfrange,modindx):
            c=pd.DataFrame([meanfreq,sd,median,Q25,Q75,skew,sp_ent,sfm,mode,meanfun,minfun,maxfun,meandom,mindom,dfrange,modindx]).T
            return model.predict(c)
          
    
    prediction=lr(meanfreq,sd,median,Q25,Q75,skew,sp_ent,sfm,mode,meanfun,minfun,maxfun,meandom,mindom,dfrange,modindx)
    
    a=''
    if prediction == 1:
        a = 'Female'
    else:
        a = 'Male'
    
    return render_template('index.html',prediction_text="The gender identified is {}".format(a))
  

if __name__=="__main__":
    app.run(debug=True)

