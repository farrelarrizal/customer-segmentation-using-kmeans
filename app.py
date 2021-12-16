from flask import Flask, render_template, request, redirect
import pickle
import sklearn
import numpy as np

app_kmeans = Flask(__name__)

@app_kmeans.route('/', methods=['POST','GET'])
def index():
        if request.method == 'POST':
                with open('kmeans_model_pickle','rb') as r:
                        model = pickle.load(r)
                        recency = int(request.form['recency'])
                        frequency = int(request.form['frequency'])
                        monetary = float(request.form['monetary'])
                        records = np.array((recency,frequency,monetary))
                        records = np.reshape(records, (1, -1))
                        hasil = model.predict(records) 
                        hasil = hasil[0]
                        print('Customer Segmentation is :',hasil)   
                
                return render_template('hasil.html',result=hasil)
        else:
                return render_template('index.html')
         
if __name__ == '__main__':
        app_kmeans.run()               