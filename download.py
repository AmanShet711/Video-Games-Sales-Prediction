from google_images_search import GoogleImagesSearch
import os
import pandas as pd
import time


data = pd.read_csv('vgsales.csv')
your_dataset_titles = data['Name']


# Function to download image for a single game title
def download_image(game_title):
    gis = GoogleImagesSearch('AIzaSyC9xSUIwNxbaOfJYlHI6tNjLWav4Uo0dmE', '74ba666ec5a6047d3')

    # Search for the game title
    _search_params = {
        'q': game_title,
        'num': 1,
        'safe': 'high',
        'fileType': 'jpg'
    }
    gis.search(search_params=_search_params)

    # Download the first image from the search results
    if gis.results():
        image = gis.results()[0]
        image_url = image.url
        image.download(f'C:/Users/Eshwar Prasad/Documents/AI_ML Mini Project/Image Downloads/{game_title.replace(":", "_")}.jpg')  # Save the image
        return image_url
# Use this function to download images for all game titles in your dataset

