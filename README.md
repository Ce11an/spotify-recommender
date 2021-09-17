# spotify-recommender
A Spotify recommendation engine that can recommend songs based on your playlist.

#### Project Status: Completed

# Table of Contents

* [Introduction](#introduction)
* [Requirements](#requirements)
* [Installation](#installation)
* [Usage](#usage)
* [References](#references)
* [License](#license)

## Project Objective
The purpose of this project was to build a simple Spotify recommender that recommends songs based on the songs within a given playlist. Also, for the recommender to add the recommended songs to the playlist.

### Requirements
* Python
* [pandas][pandas]
* [scikit-learn][scikit-learn]
* [matplotlib][matplotlib]
* [numpy][numpy]
* [regex][regex]
* [spotipy][spotipy]

## Project Description
By using songs from 1912-2020, I have built an automated recommender. To do this I transformed playlist of songs into a single vector and compare the  vector to all the songs not present in the playlist. Once I found the most similar songs, I added them to the playlist.

## Getting Started

1. Clone this repo (for help see this [tutorial][tutorial])
2. Raw Data is being kept [here][data] within this repo.
3. Recommender script is being kept [here][notebooks]
4. You will need to create a [Spotify Dashboard][dashboard] to use the recommender and connect to it via [spotipy][spotipy]

## References
* https://github.com/madhavthaker/spotify-recommendation-system [Accessed: 17 September 2021]
* https://www.linkedin.com/pulse/extracting-your-fav-playlist-info-spotifys-api-samantha-jones/ [Accessed: 17 September 2021]

[pandas]: https://pandas.pydata.org
[scikit-learn]: https://scikit-learn.org/stable/
[matplotlib]: https://matplotlib.org
[numpy]: https://numpy.org
[regex]: https://docs.python.org/3/library/re.html
[spotipy]:https://spotipy.readthedocs.io/en/2.19.0/
[tutorial]: https://help.github.com/articles/cloning-a-repository/
[data]: https://github.com/Ce11an/spotify-recommender/tree/main/data/spotify-dataset-19212020-160k-tracks
[notebooks]: https://github.com/Ce11an/spotify-recommender/tree/main/notebooks
[dashboard]: https://developer.spotify.com/dashboard/
