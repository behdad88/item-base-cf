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

movieRatings = ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')
pd.set_option('display.max_columns',10)
print(movieRatings.head(10))

# extract a Series of users who rated Star Wars
starWarsRatings = movieRatings['Star Wars (1977)']
print(starWarsRatings.head(10))

# use Pandas' corrwith function to compute the pairwise correlation of Star Wars' vector of user rating with every other movie
# After that, drop any results that have no data,
# and construct a new DataFrame of movies and their correlation score (similarity) to Star Wars
similarMovies = movieRatings.corrwith(starWarsRatings)
similarMovies = similarMovies.dropna()
df = pd.DataFrame(similarMovies)
print(df.head(10))

#sort the results by similarity score.
print(similarMovies.sort_values(ascending=False))

# getting rid of movies that were only watched by a few people that are producing spurious results.
# construct a new DataFrame that counts up how many ratings exist for each movie,
# and also the average rating
movieStats = ratings.groupby('title').agg({'rating': [np.size, np.mean]})
print(movieStats.head(10))

#get rid of any movies rated by fewer than 200 people, and check the top-rated ones that are left
popularMovies = movieStats['rating']['size'] >= 200
print(movieStats[popularMovies].sort_values([('rating', 'mean')], ascending=False)[:15])

# join this data with original set of similar movies to Star Wars
df = movieStats[popularMovies].join(pd.DataFrame(similarMovies, columns=['similarity']))
print(df.head(10))

# sort these new results by similarity score
print(df.sort_values(['similarity'], ascending=False)[:15])