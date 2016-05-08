import csv
import pickle
from sklearn import preprocessing
import numpy as np

#Load Data
csvfile = open('ml-100k/u.item')
movies = [movie for movie in csv.DictReader(csvfile, delimiter='|', fieldnames=['id','name','date','None','url'])]

csvfile = open('ml-100k/u.user')
users = [user for user in csv.DictReader(csvfile, delimiter='|')]

csvfile = open('ml-100k/u.genre')
genres = [genre['genre'] for genre in csv.DictReader(csvfile, delimiter='|', fieldnames=['genre','id'])]

csvfile = open('ml-100k/u.data')
ratings = [rating for rating in csv.DictReader(csvfile, delimiter='\t', fieldnames=['user_id','movie_id','rating','timestamp']) ]

csvfile = open('ml-100k/u.occupation')
occupations = [occupation['occupation'] for occupation in csv.DictReader(csvfile) ]

#Define age groups
age_groups = [[0,30],[30,50],[50,1000]]

#Transform to vector.
X = []
Y_age = []
Y_gender = []
for rating in ratings:
    #Get user and movie info
    user_id = rating['user_id']
    user = users[int(user_id)-1]
    movie_id = rating['movie_id']
    movie = movies[int(movie_id)-1]
    
    #Features
    x = [ int(category)*int(rating['rating']) for category in movie[None] ]    
    X.append(x)
    
    #Labels
    y_age = int(user['age'])
    y_age = [ index for index,age_group in enumerate(age_groups) if ( y_age >= age_group[0] and y_age < age_group[1] ) ][0]
    Y_age.append(y_age)
    
    y_gender = 1 if user['gender'] == 'M' else 0
    Y_gender.append(y_gender)

#Scale and persist in storage
X = preprocessing.scale(np.array(X))
Y_age = np.array(Y_age)
Y_gender = np.array(Y_gender)
pickle.dump( (X, Y_age, Y_gender), open( "data.pkl", "wb" ) )