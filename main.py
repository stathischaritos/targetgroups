import pickle
from flask import Flask, request
import numpy as np
import json

#Load models
modelAge, modelGender = pickle.load( open( "models.pkl", "rb" ) )

#API
from flask import Flask
app = Flask(__name__)

@app.route("/suggest", methods=['GET']) 
def suggest():
    input = request.args.get('input')
    
    if input:
        vec = np.array([ float(val) for val in input.split(",") ])
        response = json.dumps([ list(modelAge.predict(vec)),  list(modelGender.predict(vec)) ])
    else:
        response = "Please provide an input vector"
        
    return response

if __name__ == "__main__":
    app.run()