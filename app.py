from flask import Flask, render_template, request, redirect
import pickle
import sklearn
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['POST','GET'])
def index():
        if request.method == 'POST':
                
                with open('final_pca_pickle','rb') as c:
                        model = pickle.load(c)
                acousticness = float(request.form['acousticness'])
                danceability = float(request.form['danceability'])
                energy = float(request.form['energy'])
                instrumentalness = float(request.form['instrumentalness'])
                speechiness = float(request.form['speechiness'])
                valence = float(request.form['valence'])
                liveness = float(request.form['liveness'])
                
                datas = np.array((acousticness,danceability,energy,instrumentalness,speechiness,valence,liveness))
                datas = np.reshape(datas, (1, -1))
                pca_datas = model.transform(datas)
                
                with open('final_kmeans_pickle','rb') as r:
                        model = pickle.load(r)
                clusteringSpot = model.predict(pca_datas)    
                
                return render_template('hasil.html',finalData=clusteringSpot)
        else:
                return render_template('index.html')
         
if __name__ == '__main__':
        app.run()               