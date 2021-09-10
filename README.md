# Movie-Recommend
this is a movie-recommended system that is based on content-based. You are watching some movie this system will recommend some 5 movies 
Recommender System

what is recommender system?

    • Recommender systems are the systems that are designed to recommend things to the user based on many different factors.

    • These systems predict the most likely product that the users are most likely to purchase and are of interest to
      
    • The recommender system deals with a large volume of information present by filtering the most important information based on the data provided by a user
      

Why we use recommender syastrm?

    • Benefits users in finding items of their interest.
    • Personalized content. engagement
    • Help websites to.

Types of Recommendation System
      
    • Content-Based Recommendation System

    • Collaborative Filtering
      
    • Classification Model



Content-Based Recommendation System

    • creat tag and on the base of tag the system recommend.
      
    • works on the principle of similar content.

  e.g-
    • If a user is watching a movie, then the system will check about other movies of similar content or the same genre of the        movie the user is watching

Collaborative Filtering
    • work on the similarity between different users and also items that are widely used as an e-commerce website
      
    • User-based nearest-neighbor collaborative filtering
      
    • Item-based nearest-neighbor collaborative filtering

project flow

    • collecting all data
    • preprocessing the data (apne hisab se resdy karrege)
    • mechine learing model
    • websit convert
    • deploy

data
https://www.kaggle.com/tmdb/tmdb-movie-metadata	
tmbd 5000 movie detail

movie detail                                                                                     movie credits
genres, id, keywords, overview, title                                                 cast,crew



remove dublicate or mising data
movies.dropna(inplace = True)

convert sting into list
	import astn
		+ ast.literal_eval(text)    =     (convert sting into list)
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L

#convert into	

movies["genres"] = movies["genres"].apply(convert)
movies["keywords"] = movies["keywords"].apply(convert)

'[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]

now we have to fatching onyl three actor name

for CAST
def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L

movies["cast"] = movies["cast"].apply(convert3)

movies['cast'] = movies['cast'].apply(lambda x:x[0:3])


now we have to fatch directro from crew

for CREW

def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L

movies["crew"] = movies["crew"].apply(fetch_director)

now we have to covert OVERVIEW from string to list

OVERVIEW

movies["overview"] = movies["overview"].apply(lambda x:x.split())

now we have to remove space from “cast” “crew” “genres”  “keyword”

def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1

now we convert tags 

movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)

now we have to connotation “OVERVIEW” “GENRES” “KEYWORDS”  “CAST”  “CREW” into TAGS

movies["tags"] = movies["overview"] + movies["genres"] + movies["keywords"] + movies["cast"] + movies["crew"]

now we have to make new data frame in which there is only few colume “TITLE” “MOVIE_ID”  “TAGS”

new = movies[["movie_id",'title','tags']]


now we convert  tags list into string

new['tags'] = new['tags'].apply(lambda x: " ".join(x))

now we convert  tags value into lower    (means small leter)

new["tags"] = new["tags"].apply(lambda x : x.lower())

now we have to apply stemming function to remove all one meaning words
like 
#loved = love
#loving = love
#love = love
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
 
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new["tags"] = new["tags"].apply(stem)


now the main TEXT VECTORIZATION
          mathod BAG OF WORDS

now we use a class “from sklearn.feature_extraction.text import CountVectorizer” this help us to convert worsd into vectors

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')

stop_words = can stop all the words like is,are.the,and etc.
max_features = no of words (how many word want)

now the countvectorizer return matrix .so we have to convert into numpyarray

vector = cv.fit_transform(new['tags']).toarray()
   

now we have to calculate co-sine distance  of all movie to all movies  (har movie ka har movie ke sath distance)   (  Distance inversely proportional to 1/similarty  )

Distance = movie     distance
                     1           4806
                     2           4806
                     3           4806
                      .               .
                      .               .
                      .               .
                  4806        4806



from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vector)

now make recommended function which gives us  5 similar movies

def recommend(movie):
    movie_index = new[new['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)),reverse=True,key = lambda x: x[1])[1:6]
    
    for i in movie_list:
        print(new.iloc[i[0]].title)
    
now we have to make a web site


import pandas as pd
import streamlit as st 
import pickle


def recommend(movie):
    movie_index = movies[movies["title"] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    for i in movies_list:
        # movie_id = i[0]
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies


movies_dict = pickle.load(open("movies_dict.pkl", "rb"))
movies = pd.DataFrame(movies_dict)

similarity = pickle.load(open("similarity.pkl", "rb"))

st.title("Movie Recommender System")

selected_Movie_Name = st.selectbox(
    'How would you like to be contacted?',
    movies["title"].values)

if st.button('Recommend'):
    recommendations = recommend(selected_Movie_Name)
    for i in recommendations:
        st.write(i)
