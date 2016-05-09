import csv
import pickle
from sklearn import preprocessing
import numpy as np
from sklearn.decomposition import PCA

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
age_groups = [[0,30],[30,100]]
N_COMP = 9

#Transform to vector.
X = []
Y_age = []
Y_gender = []
count1 = 0
count2 = 0
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
    
    y_gender = 0 if user['gender'] == 'M' else 1
    Y_gender.append(y_gender)
    

#reduce dimensions and persist in storage
X = np.array(X)
X = preprocessing.scale(X)
pca = PCA(n_components=N_COMP, whiten=True)
pca.fit(X)
X = pca.transform(X)

Y_age = np.array(Y_age)
Y_gender = np.array(Y_gender)

pickle.dump( (X, Y_age, Y_gender, pca), open( "data.pkl", "wb" ) )