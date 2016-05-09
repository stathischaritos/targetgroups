import pickle
from flask import Flask, request
import numpy as np
import json
from sklearn.neighbors import KNeighborsClassifier

#Load models
modelAge, modelGender, pca = pickle.load( open( "models.pkl", "rb" ) )

#API
from flask import Flask
app = Flask(__name__)

@app.route("/suggest", methods=['GET']) 
def suggest():
    input = request.args.get('input')
    
    if input:
        vec = pca.transform(np.array([ float(val) for val in input.split(",") ]))
        response = json.dumps([ list(modelAge.predict_proba(vec)[0]),  list(modelGender.predict_proba(vec)[0]) ])
    else:
        response = "Please provide an input vector"
        
    return response

if __name__ == "__main__":
    app.run()