# spotify-recommender
A Spotify recommendation engine that can recommend songs based on your playlist.

#### -- Project Status: Completed

## Project Objective
The purpose of this project was to build a simple Spotify recommender that recommends songs based on the songs within a given playlist. Also, for the recommender to add the recommended songs to the playlist.

### Technologies
* Python
* pandas, jupyter, scikit-learn, regex, spotipy, matplotlib

## Project Description
By using songs from 1912-2020, I have built an automated recommender. To do this I transformed playlist of songs into a single vector and compare the  vector to all the songs not present in the playlist. Once I found the most similar songs, I added them to the playlist.

## Getting Started

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Raw Data is being kept [here](https://github.com/Ce11an/spotify-recommender/tree/main/data/spotify-dataset-19212020-160k-tracks) within this repo.
3. Recommender script is being kept [here](https://github.com/Ce11an/spotify-recommender/tree/main/notebooks)
4. You will need to create a [Spotify Dashboard] (https://developer.spotify.com/dashboard/) to use the recommender and connect to it via [spotipy](https://spotipy.readthedocs.io/en/2.19.0/)

## References
* https://github.com/madhavthaker/spotify-recommendation-system
* https://www.linkedin.com/pulse/extracting-your-fav-playlist-info-spotifys-api-samantha-jones/
