import pandas as pd
import numpy as np

# loading up the MovieLens dataset

r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('u.data', sep='\t', names=r_cols, usecols=range(3), encoding="ISO-8859-1")

m_cols = ['movie_id', 'title']
movies = pd.read_csv('u.item', sep='|', names=m_cols, usecols=range(2), encoding="ISO-8859-1")

ratings = pd.merge(movies, ratings)
print(ratings.head(10))

#using pivot_table function on a DataFrame to construct a user / movie rating matrix.
#NaN indicates missing data - movies that specific users didn't rate.

userRatings = ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')
pd.set_option('display.max_columns',10)
print(userRatings.head(10))

# using pandas built-in corr() method to compute a correlation score for every column pair in the matrix
# This gives us a correlation score between every pair of movies (where at least one user rated both movies - otherwise NaN's will show up)
# to avoid spurious results that happened from just a handful of users that happened to rate the same pair of movies
# and to restrict our results to movies that lots of people rated together
# we'll use the min_periods argument to throw out results where fewer than 100 users rated a given movie pair
corrMatrix = userRatings.corr(method='pearson', min_periods=100)
print(corrMatrix.head())

# Now produce some movie recommendations for user ID 0,
# This guy really likes Star Wars and The Empire Strikes Back, but hated Gone with the Wind.
# extract his ratings from the userRatings DataFrame, and use dropna() to get rid of missing data

personRatings = userRatings.loc[0].dropna()
print(personRatings)

simCandidates = pd.Series()
for i in range(0, len(personRatings.index)):
    print("Adding sims for " + personRatings.index[i] + "...")
    # Retrieve similar movies to this one that I rated
    sims = corrMatrix[personRatings.index[i]].dropna()
    # Now scale its similarity by how well I rated this movie
    sims = sims.map(lambda x: x * personRatings[i])
    # Add the score to the list of similarity candidates
    simCandidates = simCandidates.append(sims)

# Glance at our results so far:
print("sorting...")
simCandidates.sort_values(inplace=True, ascending=False)
print(simCandidates.head(10))

# some of the same movies came up more than once,
# because they were similar to more than one movie rated.
# We'll use groupby() to add together the scores from movies that show up more than once,
#  so they'll count more

simCandidates = simCandidates.groupby(simCandidates.index).sum()
simCandidates.sort_values(inplace = True, ascending = False)
print(simCandidates.head(10))

# filter out movies person has already rated
filteredSims = simCandidates.drop(personRatings.index)
print(filteredSims.head(10))