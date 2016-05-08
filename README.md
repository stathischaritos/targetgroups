README
======
This project uses the movielens dataset to train an recomender system. The system takes as input movie category weights and outputs an age group and gender prediction. A use case could be movie marketers who
want to know where to target their campaigns for a certain film (ex. in promoted facebook pages you can target certain groups).

DEPENDENCIES
============

- scikit
- scikit-neuralnetwork
- numpy
- flask

SETUP
=====
You need to run data.py to preprocess the data and train.py to train the models. Finaly main.py runs the microservice.

USAGE
=====
The models need an input vector of weights for each movie category/genre. The genres are all inside u.genre. 

example(random):

    http://localhost:5000/suggest?input=0.1,0.2,0.5,0.0,1.0,0.2,0.0,0.0,0.0,0.2,0.1,0.2,0.5,0.0,1.0,0.2,0.0,0.0,0.0