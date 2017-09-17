# Recommender-By-Matrix-Factorization


## Collaborative filtering
This kind of task is also known as **collaborative filtering**, and lies in the **unsupervised leraning** category. In our case, movie recommendation: we concern to predict which movies people will want to watch based on how they, and other people, have rated movies which they have already seen. The key idea is that the prediction is not based on features of the movie or user (although it could be), but merely on a ratings matrix. More precisely, we have a matrix **X** where **X**(m,u) is the (here an integer between 1 and 5, where 1 is dislike and 5 is like) by user *u* of movie *m*. Note that most of the entries in **X** will be missing or unknown, since most users will not have rated most movies. Hence we only observe a tiny subset of the **X** matrix, and we want to predict a different subset. In particular, for any given user *u*, we might want to predict which of the unrated movies he/she is most likely to want to watch. rating

## Task description and result
This is a Movie Recommender System Based on Matrix Factorization, train and test on the *movielens100k.mat* dataset. The dataset for this project comes from MovieLens (http://grouplens.org/). It consists of 100,000 ratings from 1000 users on 1700 movies. The rating values go from 0 to 5. The dataset is split into train and test data, around 95,000 and 5,000 ratings respectively.

With all the parameters well set, a RMSE of around 0.8008 (actually MSE according to the formula provided) can be achieved.


## References
- Murphy, Kevin P. Machine learning: a probabilistic perspective. MIT press, 2012.
