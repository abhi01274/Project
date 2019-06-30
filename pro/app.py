from flask import Flask,render_template, request
import pickle
import numpy as np
import pandas as pd






app = Flask(__name__)


@app.route('/')
def hello_world():
    data=pd.read_excel('project/project/fsi-2019.xlsx')
    country_list=data.iloc[:,0]

    return render_template("index.html",country_list=country_list)


@app.route('/home',methods=['POST'])
def result():

    form_data=request.form
    country=form_data["country"].lower()
    SA=form_data["SA"]
    FE=form_data["FE"]
    GG=form_data['GG']
    ECO=form_data['ECO']
    ECO_EE=form_data['ECO_EE']
    HFBD=form_data['HFBD']
    SL=form_data['SL']
    PS=form_data['PS']
    HR=form_data['HR']
    DP=form_data['DP']
    RIDP=form_data['RIDP']
    XI=form_data['XI']
    year=form_data['year']


    lst=[ country , SA , FE , GG , ECO , ECO_EE , HFBD , SL , PS , HR , DP , RIDP , XI , year]
    lst=np.array(lst)
    lst=lst.reshape(1,-1)

    with open('static/pickle/model_le.pkl', 'rb') as fp:
        label_enco = pickle.load(fp)
        lst[:,0]=label_enco.transform(lst[:,0])

    with open('static/pickle/model_ohe.pkl', 'rb') as fp1:
        OHE = pickle.load(fp1)
        lst=OHE.transform(lst).toarray()
        lst=lst[:,1:]

    with open('static/pickle/model_sc.pkl', 'rb') as fp2:
        SC = pickle.load(fp2)
        lst=SC.transform(lst)
    with open('static/pickle/classifier.pkl', 'rb') as fp3:
        model = pickle.load(fp3)
        result=model.predict(lst)



    if result==0:
        ans="Moderate state"
    elif result==1:
        ans="warning state"
    elif result==2:
        ans="Risky"
    #elif result==3:
        #ans='Highly Risky'
    else:
        ans=" Invalid Inputs . Kindly Retry"


    info="Country "+ country+" is "+ans
    data=pd.read_excel('project/project/fsi-2019.xlsx')
    country_list=data.iloc[:,0]


    return render_template("index.html", info=info,country_list=country_list)





if __name__ == '__main__':
    app.run(debug=True)
