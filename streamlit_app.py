from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
from flask import Flask

if not hasattr(st, 'already_started_server'):
    # Hack the fact that Python modules (like st) only load once to
    # keep track of whether this file already ran.
    st.already_started_server = True

    st.write('''
        The first time this script executes it will run forever because it's
        running a Flask server.

        Just close this browser tab and open a new one to see your Streamlit
        app.
    ''')

    app = Flask(__name__)

    @app.route('/foo')
    def serve_foo():
        return 'This page is served via Flask!'

    app.run(port=8888,debug=True)

def predict_quality(model, df):
    predictions_data = predict_model(estimator=model, data=df)
    return predictions_data['Label'][0]

#model = load_model('bolig_algo')

data = [
        5,#float(pricedev),
        "d",#str(energymark),
        "2200",#str(postal),
        3500,#float(expense),
        82,#areaResidential,
        4,#float(numRooms),
        12,#float(salesPeriod),
        "Home",#str(agent),
        0,#etage,
        1,#disToCph,
        1,#disToFbr,
        1,#disToØst,
        1,#distances_list[0],#cafe
        1,#distances_list[1],#station
        1,#distances_list[2],#subway
        1,#distances_list[3],#restaurant
        1,#distances_list[4],#pharmacy
        1,#distances_list[5],#school
        1,#distances_list[6],#bar
        1,#distances_list[7],#pub
        ]

columns = ["priceDevelopment","energyMark","postal","paymentExpenses","areaResidential","numberOfRooms","salesPeriod","agentChainName"
           ,"stue","disToCph","disToFbr","disToØst","cafe","station","subway","restaurant","pharmacy","school","bar","pub"]
newData = pd.DataFrame([data],columns=columns)
#result = predict_quality(model,newData)


#st.title('bolig predicter')


#st.table(newData)

#if st.button('Predict'):
#    prediction = predict_quality(model, newData)
#
#    st.write(' Based on feature values, your wine quality is ' + str(prediction))