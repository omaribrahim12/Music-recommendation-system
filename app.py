#import libraries
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict
import numpy as np
import pickle
import streamlit as st
import pandas as pd
from collections import defaultdict
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
import difflib


#loading the saved model
data = pd.read_csv("C:/Users/Dell/Desktop/final-project2/data.csv")
loaded_model = pickle.load(open('C:/Users/Dell/Desktop/final-project2/train_model.sav','rb'))



sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="d464d11a89a243abb401de7e8697a0b3",
                                                           client_secret="4351f166f1c246c4afa614bc83fecc11"))
#function 1 

def song_search(name, year):
    song_info = defaultdict()
    results = sp.search(q= 'track: {} year: {}'.format(name,year), limit=1)
    if results['tracks']['items'] == []:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_info['name'] = [name]
    song_info['year'] = [year]
    song_info['explicit'] = [int(results['explicit'])]
    song_info['duration_ms'] = [results['duration_ms']]
    song_info['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_info[key] = value

    return pd.DataFrame(song_info)


#for example this one 
song_search('levitating (feat. dababy)',2020 )
song_search('Que Fue Lo Que Paso',2020) 


#function 2 

number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']


def get_song_data(song, spotify_data):
    
    try:
        song_info = spotify_data[(spotify_data['name'] == song['name']) 
                                & (spotify_data['year'] == song['year'])].iloc[0]
        return song_info
    
    except IndexError:
        return find_song(song['name'], song['year'])
    
def get_mean_vector(song_list, spotify_data):
    
    song_vectors = []
    
    for song in song_list:
        song_info = get_song_data(song, spotify_data)
        if song_info is None:
            print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
            continue
        song_vector = song_info[number_cols].values
        song_vectors.append(song_vector)  
    
    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)


def flatten_dict_list(dict_list):
    
    flattened_dict = defaultdict() #generate an empty dict 
    for key in dict_list[0].keys():#add the names of every single songs 
        flattened_dict[key] = []
    
    for dictionary in dict_list: # add the years (values) to the 
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
            
    return flattened_dict

def recommend_songs( song_list, spotify_data, n_songs=10):
    
    metadata_cols = ['name','artists']
    song_dict = flatten_dict_list(song_list)
    
    song_center = get_mean_vector(song_list, spotify_data)
    scaler = loaded_model.steps[0][1]
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])
    
    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    
    return rec_songs[metadata_cols].to_dict(orient='records')

#test
input_data =[{'name': 'Once Upon a December', 'year':2016}]




import time
def main():
    

    #giving a title 
    st.title('Music recommendation system')
    

    name = data['name'].unique()
    song_name =st.selectbox('Enter your favourite song!',name)
    data1 = data[data['name'] == song_name][['name']+['artists']+['year']]
    st.dataframe(data1,width=700)
    year = data['year'].unique()
    song_year =st.selectbox('Enter the year of release',year)
   
    #code for predection 
    result = [] #to save the outut arrays 
    
    #create button for Prediction 
    
    if st.button('Predict the song! '):
        
        #append the function output into result (array)
        result = recommend_songs([{'name': song_name, 'year':song_year}],  data )
        
        #inform the user to wait for mins until the result be ready
        with st.spinner('Wait for it...'):
            time.sleep(3)
            
        #Add text to inform the users that the list is ready 
        st.header("This is your top 10 songs list!!") 
        st.header("Here is your song names and artists!") 
        #for loop to extract the result into different lines and make it clear to the user 
        for x in range (9):               
            st.markdown(result[x])
        st.success('Done!')
        st.balloons()
        
              
    
 

    ## Search for your favourite artist!
    # artist_search = data['artists'].unique()
    artists = data['artists'].unique()
    artists=st.selectbox('Enter your favourite singer!',artists )
    st.markdown("Your selection is "+artists+"!")
    data[data['artists'] == artists][['name']+['year']+['popularity']]
    
    
if __name__ == '__main__':
    main()





















