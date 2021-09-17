import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
import spotipy.util as util
from skimage import io
import matplotlib.pyplot as plt


def call_playlist(username, playlist_id):
    """
    Obtaining song information from a playlist using the Spotify API.

    Parameters
    ----------
    username: str - spotify username.
    playlist_id: str - Can be obtained from the playlist url.
    
    Returns
    -------
    A DataFrame containing the songs from the specified playlist.
    
    Examples
    --------
    call_playlist(playlist_id="playlist_id1cOdetiuvPeITyxYfBYQSP")
    """
    
    playlist_features_list = [
        "artist",
        "album", 
        "song_name",
        "id",
        "cover_url",
        "date_added",
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode", 
        "speechiness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo", 
        "duration_ms",
        "time_signature"
        
    ]
    
    df_playlist = pd.DataFrame(columns = playlist_features_list)
    
    playlist = sp.user_playlist_tracks(username, playlist_id)["items"]
    for song in playlist:
        playlist_features = {}
        playlist_features["artist"] = song["track"]["album"]["artists"][0]["name"]
        playlist_features["album"] = song["track"]["album"]["name"]
        playlist_features["song_name"] = song["track"]["name"]
        playlist_features["id"] = song["track"]["id"]
        playlist_features["cover_url"] = song["track"]["album"]['images'][1]['url']
        playlist_features["date_added"] = pd.to_datetime(song["added_at"])
        
        audio_features = sp.audio_features(playlist_features["id"])[0]
        for feature in playlist_features_list[6:]:
            playlist_features[feature] = audio_features[feature]
        
        df_song = pd.DataFrame(playlist_features, index = [0])
        df_playlist = pd.concat([df_playlist, df_song], ignore_index = True)
        
    return df_playlist

def visualise_songs(df):
    """
    Show the album cover for each song.

    Parameters
    ----------
    df: pandas.DataFrame - containing the url for the album cover and song name.
    
    Returns
    -------
    Matplotlib plot showing the album cover for each song.
    
    Examples
    --------
    visualise_songs(df=df_playlist)
    """
    plt.figure(figsize=(15,int(0.625 * len(df['cover_url']))))
    columns = 3
    
    for i, url in enumerate(df['cover_url']):
        plt.subplot(int(len(df['cover_url']) / columns + 1), columns, i + 1)

        image = io.imread(url)
        plt.imshow(image)
        plt.xticks(fontsize = 0.1)
        plt.yticks(fontsize = 0.1)
        plt.xlabel(df['song_name'].values[i], fontsize = 12)
        plt.tight_layout(h_pad=0.4, w_pad=0)
        plt.subplots_adjust(wspace=None, hspace=None)

    return plt.show()

def playlist_features(df_playlist, df_song_feat, weight):
    """
    Transforms playlist songs into features for similarity comparison.
    
    Parameters
    ----------
    df_playlist: pandas.DataFrame - a playlist of songs.
    
    df_song_feat: pandas.DataFrame - all spotify songs from 1912-2020 transformed into features.
    
    weight: float (0-1) - to account for recency bias. The larger the value the more weight is added to the most recent songs added to the playlist. 
    
    Returns
    -------
    Two pandas.Dataframe(s) 
    
    return 1 - a single vector representation of the playlist.
    return 2 - all the songs that are not in the playlist.
    
    Examples
    --------
    playlist_features(df_playlist=df_playlist, df_song_feat=df_song_feat, weight=1)
    """
    df_playlist_feat = pd.merge(
        df_song_feat[df_song_feat['id'].isin(df_playlist['id'])],
        df_playlist[['id', 'date_added']],
        how='inner',
        on='id'
    ).sort_values('date_added',ascending=False).reset_index(drop=True)

    df_non_playlist_feat = df_song_feat[~df_song_feat['id'].isin(df_playlist['id'])].reset_index(drop=True)

    basline = df_playlist_feat['date_added'][0]
    df_playlist_feat['days_since_first'] = df_playlist_feat['date_added'].apply(lambda x: (basline - x).days)
    df_playlist_feat['weight'] = df_playlist_feat['days_since_first'].apply(lambda x: weight ** (-x))
    df_playlist_weight = pd.DataFrame(df_playlist_feat.iloc[:,1:-3].mul(df_playlist_feat['weight'], 0).mean(axis=0)).transpose()

    return df_playlist_weight, df_non_playlist_feat


df_song = pd.read_csv('spotify-dataset-19212020-160k-tracks/data.csv')

print(df_song.info())

df_song.head()


df_w_genres = pd.read_csv('spotify-dataset-19212020-160k-tracks/data_w_genres.csv').rename(columns={"artists":"genre_artists"})

print(df_w_genres.info())

df_w_genres.head()


df_song['artists_clean'] = df_song['artists'].apply(lambda x: re.findall(r"(\b(?get_ipython().getoutput("\')[^\,\[\]]+(?<!\')\b)", x))")


df_w_genres['genres_clean'] = df_w_genres['genres'].apply(lambda x: re.findall(r"(\b(?get_ipython().getoutput("\')[^\,\[\]]+(?<!\')\b)", x))")
df_w_genres['genres_clean'] = df_w_genres['genres_clean'].apply(lambda x: [string.replace(" ", "_") for string in x])

df_w_genres['genres_clean'] = np.where(
    df_w_genres['genres'] == '[]', 
    np.nan,
    df_w_genres['genres_clean']
)


df_song['song_artist'] = df_song.apply(lambda x: x['artists_clean'][0]+'_'+x['name'], axis=1)


df_song = df_song.sort_values(['song_artist','release_date'], ascending=True)
df_song = df_song.drop_duplicates('song_artist', keep='last').reset_index(drop=True)


df_link = pd.merge(
    df_song.explode('artists_clean'),
    df_w_genres[['genre_artists', 'genres_clean']],
    how='left',
    left_on='artists_clean',
    right_on='genre_artists',
).drop(['genre_artists'], axis=1).reset_index(drop=True)

for row in df_link.loc[df_link['genres_clean'].isnull(), 'genres_clean'].index:
    df_link.at[row, 'genres_clean'] = []


df_link['genres_comb'] = df_link.groupby('id')['genres_clean'].transform('sum')
df_link['genres_comb'] = df_link['genres_comb'].apply(lambda x: list(set(x)))

df_link['genres_comb'] = df_link['genres_comb'].apply(lambda x: " ".join(x))


cols_int = list(df_link.dtypes[df_link.dtypes=='int64'].index.values)
cols_int.remove('duration_ms')
df_int = df_link[cols_int]
df_ohe = pd.get_dummies(data=df_int, columns=cols_int)


cols_float = df_link.dtypes[df_link.dtypes=='float64'].index.values
df_float = df_link[cols_float]

scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_float), columns = df_float.columns)


tfidf = TfidfVectorizer()
tfidf_matrix =  tfidf.fit_transform(df_link['genres_comb'])
df_genre = pd.DataFrame(tfidf_matrix.toarray())
df_genre.columns = ['genre' + "|" + i for i in tfidf.get_feature_names()]


df_song_feat = pd.concat([df_link['id'], df_ohe, df_scaled, df_genre], axis=1)


CLIENT_ID = 'CLIENT_ID'
CLIENT_SECRET = 'CLIENT_SECRET'
SPOTIFY_USERNAME = 'SPOTIFY_USERNAME'

scope = 'playlist-modify-public'
token = util.prompt_for_user_token(
    SPOTIFY_USERNAME, 
    scope, 
    client_id=CLIENT_ID, 
    client_secret=CLIENT_SECRET,
    redirect_uri='http://localhost/'
) 

sp = spotipy.Spotify(auth=token)


PLAYLIST_ID = 'PLAYLIST_ID'

df_playlist = call_playlist(username=SPOTIFY_USERNAME, playlist_id=PLAYLIST_ID)

df_playlist.head()


visualise_songs(df_playlist)


df_playlist_weight, df_non_playlist_feat = playlist_features(df_playlist=df_playlist, df_song_feat=df_song_feat, weight=1)


df_non_playlist_feat['song_similarity'] = cosine_similarity(df_non_playlist_feat.drop('id', axis = 1), df_playlist_weight)


non_playlist_top_5 = df_non_playlist_feat.sort_values('song_similarity', ascending = False).head(5)

top_5 = pd.merge(
    non_playlist_top_5,
    df_song,
    on='id',
    how='inner'
).rename(columns={"name":"song_name"})

top_5['cover_url'] = top_5['id'].apply(lambda x: sp.track(x)['album']['images'][1]['url'])


visualise_songs(top_5)


sp.user_playlist_add_tracks(SPOTIFY_USERNAME, PLAYLIST_ID, list(top_5['id']))
