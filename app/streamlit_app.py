import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import random
from datetime import datetime
from streamlit_gsheets import GSheetsConnection

# Print the songs from the given artist
def print_songs_from_artist(artist_name):
    print(df[df["artist_name"].str.lower() == artist_name.lower()][["track_name", "artist_name"]].values.tolist())

# Get the average features of the given artist
def get_features_by_artist(artist_name):
    artist_features = df[df["artist_name"].str.lower() == artist_name.lower()].iloc[:, 3:].mean().values.reshape(1, -1)
    return artist_features

# Get the features of the given song
def get_features_by_track_id(track_id):
    features = df[df["track_id"] == track_id].iloc[:, 3:].values
    return features

# Get average features of the given songs
def get_features_by_track_ids(track_ids):
    #print(track_ids)
    #features = df[df["track_id"].isin(track_ids)].iloc[:, 3:].values
    features = df[df["track_id"].isin(track_ids)].iloc[:, 3:].mean().values.reshape(1, -1)
    return features

# Get average features of the given songs and artists
def get_features_by_artists_and_track_ids(artists, track_ids):
    artist_features = []
    for artist in artists:
        artist_features.append(get_features_by_artist(artist))
    song_features = []
    for track_id in track_ids:
        song_features.append(get_features_by_track_id(track_id))
    
    # append the song features to the artist features even if one is empty
    if artist_features and song_features:
        features = np.concatenate((np.concatenate(artist_features), np.concatenate(song_features)), axis=0)
    elif artist_features:
        features = np.concatenate(artist_features, axis=0)
    elif song_features:
        features = np.concatenate(song_features, axis=0)
    else:
        raise ValueError("Both artist_features and song_features are empty")
    # calculate the mean of the features
    features = np.mean(features, axis=0).reshape(1, -1)

    return features

# Get artists with similar features
def search_artists_by_features(features, n=5):
    #print(features)
    df_artists = df.groupby("artist_name").agg({col: "mean" for col in df.select_dtypes(include=np.number).columns})
    distances = cdist(features, df_artists.values, metric="euclidean")[0]
    similar_artist_indices = distances.argsort()[0:n]
    similar_artist_names = df_artists.iloc[similar_artist_indices].index.tolist()
    return similar_artist_names

# Get songs with similar features
def search_songs_by_features(features, n=5):
    distances = cdist(features, df.iloc[:, 3:].values, metric="euclidean")[0]
    similar_song_indices = distances.argsort()[0:n]
    similar_song_ids = df.iloc[similar_song_indices]["track_id"].tolist()
    similar_artist_names = df.iloc[similar_song_indices]["artist_name"].tolist()
    return [similar_artist_names, similar_song_ids]

# Get the artist of the given song
def get_artist_by_track_id(track_id):
    artist = df[df["track_id"] == track_id]["artist_name"].values[0]
    return artist

# Return artists and songs that rank highly in the list of similar songs
# Receives a list of a hundred song ids and gives them a score based on their position in the list
# The score is calculated as 100 - position in the list
# Calculates the total score for each artist based on their top 4 songs in the list
# Returns the top 5 artists and their top 4 songs
def generate_recommendations(input_artists, features, randomness = 1):
    n = 100
    if randomness > 2:
        n = n * (randomness - 1)
    similar = search_songs_by_features(features, n=n)
    song_artists = similar[0]
    similar_ids = similar[1]

    # Create dictionary that maps each song ID to its index in the similar_ids list
    id_to_index = {song_id: i for i, song_id in enumerate(similar_ids)}
    
    # Calculate scores for each song based on position in the list
    scores = [n - i for i in range(len(similar_ids))]

    # Shuffle the scores to add randomness based on the randomness parameter
    if randomness > 1:
        random.shuffle(scores)

    # Group songs by artist
    artist_songs = {}
    for i, artist in enumerate(song_artists):
        if artist not in artist_songs:
            artist_songs[artist] = []
        artist_songs[artist].append(similar_ids[i])
    
    # Calculate total score for each artist based on top 4 songs
    artist_scores = {}
    artist_song_counts = {}
    for artist, songs in artist_songs.items():
        top_songs = sorted(songs, key=lambda x: scores[id_to_index[x]], reverse=True)[:4]
        artist_scores[artist] = sum([scores[id_to_index[song_id]] for song_id in top_songs])
        artist_song_counts[artist] = sum([1 for song_id in top_songs if song_id in similar_ids])
    
    # Sort artists by score and return top 4 songs for each
    sorted_artists = sorted(artist_scores.items(), key=lambda x: x[1], reverse=True)
    recommendations = {}
    threshold = 100 + (n - 100) * (1200 - 100) / (500 - 100)
    #st.write(f"Threshold: {threshold}")
    for artist, score in sorted_artists:
        if artist_song_counts[artist] >= 2 and score > threshold and artist not in input_artists:
            top_songs = sorted(artist_songs[artist], key=lambda x: scores[id_to_index[x]], reverse=True)[:4]
            recommendations[artist] = top_songs

    # Print point values for each recommended artist and song
    #for artist, songs in recommendations.items():
    #    st.write(f"{artist}: {artist_scores[artist]}")
    #    st.write([scores[id_to_index[song_id]] for song_id in songs])

    return recommendations

st.set_page_config(page_title='Vibe - Music Recommendation System',
                   initial_sidebar_state=st.session_state.get('sidebar_state', 'expanded'))

cols = 2

#import os
#print(os.listdir())

import base64

with open("app/Vibe Wide Cropped.png", "rb") as f:
    data = base64.b64encode(f.read()).decode("utf-8")

    st.sidebar.markdown(
        f"""
        <div style="display:table;margin-top:-28%;margin-left:-9%;margin-right:-10%;margin-bottom:10%">
            <img src="data:image/png;base64,{data}" width="336" height="168">
        </div>
        """,
        unsafe_allow_html=True,
    )

# Add logo wide
#st.sidebar.image('Vibe Wide.png')

# Create expandable pane for settings
with st.sidebar.expander('Settings', expanded=False):
    #add slider for number of columns
    #cols = st.slider('Number of columns', 1, 4, 3)
    max_artists = st.slider('Maximum number of recommended artists', 1, 6, 4)
    randomness = st.slider('Variance', 1, 5, 2, help='This parameter increses the variance of recommendations. The lowest setting of 1 will always give the same recommendations for identical inputs. Higher values will give more diverse recommendations.')

@st.cache_data
def load_data():
    return pd.read_parquet('data/data_encoded.parquet')

df = load_data()

# Create multiselect for artist selection
artist_names = df['artist_name'].unique()
artists = st.sidebar.multiselect('Select up to 5 artists', artist_names, default=[], key='artists', max_selections=5)

# Create song selection for each artist
track_ids = []
selection_dict = {}
for artist in artists:
    selection_dict[artist] = []
    st.sidebar.write(f"### {artist}")
    track_names = df[df['artist_name']==artist]['track_name'].unique()
    selected_tracks = st.sidebar.multiselect('Select specific songs', track_names, default=[], key=artist, max_selections=3)
    for track_name in selected_tracks:
        if track_name != '':
            selection_dict[artist].append(track_name)
            # Check that artist and track name are unique
            track_id = df[(df['artist_name']==artist) & (df['track_name']==track_name)]['track_id'].unique()[0]
            track_ids.append(track_id)


def save_recommendations(artist_dict, recommendations):
    # Save recommendations as an HTML file
    html = """
    <html>
    <head>
        <title>Vibe Recommendations</title>
        <style>
            body {
                font-family: sans-serif;
            }
            iframe {
                margin-bottom: 10px;
            }
        </style>
    </head>
    <body>
    """
    html += "<h1>Input Artists:</h1>"
    html += ', '.join([f"{artist} ({', '.join(songs)})" for artist, songs in artist_dict.items()])
    
    html += "<h1>Recommendations:</h1>"
    for artist, songs in recommendations.items():
        html += f"<h2>{artist}</h2>"
        for song_id in songs:
            html += f'<iframe src="https://open.spotify.com/embed/track/{song_id}" width="275" height="80" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>'
            html += "   "
    html += "</body></html>"
    return html

def save_input_artists(artists_songs):
    conn = st.connection("gsheets", type=GSheetsConnection, ttl=0)
    sheet = conn.read(worksheet="Data", usecols=list(range(21)), ttl=0)

    # Create a new row with the current date and time
    new_row = [datetime.now().strftime("%Y%m%d-%H%M%S")]

    # Add the artists and songs to the new row
    for artist, songs in artists_songs.items():
        new_row += [artist] + songs + [None] * (3 - len(songs))

    # Pad new_row with None for any missing columns
    new_row += [None] * (len(sheet.columns) - len(new_row))

    # Convert new_row to a DataFrame
    new_row_df = pd.DataFrame([new_row], columns=sheet.columns)

    # Find the first non-NaN row
    sheet = sheet.dropna(how="all")

    sheet = pd.concat([sheet, new_row_df], ignore_index=False)
    conn.update(worksheet="Data", data=sheet)

if len(artists) > 0:
    #add button to search for similar artists and songs
    if st.sidebar.button('Search for similar artists'):
        if len(artists) == 0:
            st.error("Please select at least one artist")
        else:
            features = get_features_by_artists_and_track_ids(artists, track_ids)
            recommendations = generate_recommendations(artists, features, randomness)
            recommendations = {k: recommendations[k] for k in list(recommendations)[:max_artists]}

            # Display songs in two columns
            col_list = st.columns(cols)
            for i, (artist, songs) in enumerate(recommendations.items()):
                col_index = i % cols
                with col_list[col_index]:
                    st.write("### " + artist.replace('$', '\$'))
                    for song_id in songs:
                        st.write(f'<iframe src="https://open.spotify.com/embed/track/{song_id}" width="300" height="80" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>', unsafe_allow_html=True)
                    #st.divider()
                    st.text("")

            # Save recommendations as a file
            st.download_button('Download recommendations', save_recommendations(selection_dict, recommendations), file_name='recommendations' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.html')

            save_input_artists(selection_dict)

    else:
        st.header('Vibe - Music Recommendation System')
        st.divider()
        #instructions to search
        st.write('#### Add more artists or select specific songs to tune recommendations')
        st.write('#### Click the search button to generate recommendations')
else:
    st.header('Vibe - Music Recommendation System')
    st.divider()

    #instructions to search
    st.write('#### Select at least one artist to generate recommendations')


