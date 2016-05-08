import pickle
from sklearn import preprocessing, cross_validation
import numpy as np
from sklearn import linear_model

#Load preprocessed data
X, Y_age, Y_gender = pickle.load( open( "data.pkl", "rb" ) )

#Initialize and train models
modelAge = linear_model.LogisticRegression(C=1e5)
modelGender = linear_model.LogisticRegression()
modelAge.fit(X, Y_age)
modelGender.fit(X, Y_gender)

#Persist in storage
pickle.dump( (modelAge, modelGender), open( "models.pkl", "wb" ) )

#Cross validation to check accuracy
#print np.mean(cross_validation.cross_val_score(modelAge, X, Y_age, scoring='accuracy'))
#print np.mean(cross_validation.cross_val_score(modelGender, X, Y_gender, scoring='accuracy'))

