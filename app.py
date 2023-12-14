from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import io
import urllib, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import requests
from urllib.parse import quote

import time

import os

from download import download_image

app = Flask(__name__, static_url_path='/static')

# Load and preprocess the dataset
data = pd.read_csv("vgsales.csv")
data = data.dropna()

# Prepare the model
x = data[["Rank", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]]
y = data["Global_Sales"]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        user_input = {
            'Rank': float(request.form['Rank']),
            'NA_Sales': float(request.form['NA_Sales']),
            'EU_Sales': float(request.form['EU_Sales']),
            'JP_Sales': float(request.form['JP_Sales']),
            'Other_Sales': float(request.form['Other_Sales'])
        }
        user_data = pd.DataFrame([user_input])
        prediction = model.predict(user_data)[0]
        # Find the game closest to the predicted value
        closest_game = data.iloc[(data['Global_Sales'] - prediction).abs().argsort()[:1]]
        closest_game_name = closest_game['Name'].values[0]
        image_path = download_image(closest_game_name)
        return redirect(url_for('results', prediction=prediction, closest_game_name=closest_game_name, image_url=image_path))


    return render_template('predict.html', prediction=None)

@app.route('/results/<prediction>/<closest_game_name>/<path:image_url>')
def results(prediction, closest_game_name, image_url):
    return render_template('results.html', prediction=prediction, closest_game_name=closest_game_name, image_url=quote(image_url, safe=':/'))
@app.route('/insights')
def insights():
    # Generating insights from the dataset
    genre_sales = data.groupby("Genre")["Global_Sales"].sum().nlargest(10)
    platform_sales = data.groupby("Platform")["Global_Sales"].sum().nlargest(10)

    # Pie chart for Top 10 Genres of Games Sold
    game = data.groupby("Genre")["Global_Sales"].count().head(10)
    custom_colors = mcolors.Normalize(vmin=min(game), vmax=max(game))
    colours = [plt.cm.PuBu(custom_colors(i)) for i in game]
    plt.figure(figsize=(7,7))
    plt.pie(game, labels=game.index, colors=colours)
    central_circle = plt.Circle((0, 0), 0.5, color='white')
    fig = plt.gcf()
    fig.gca().add_artist(central_circle) 
    plt.rc('font', size=12)
    plt.title("Top 10 Genres of Games Sold", fontsize=20)
    
    # Save the pie chart to a bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(8, 6))
    corr = data.corr()
    sns.heatmap(corr, cmap="YlOrBr")
    
    # Save the heatmap to a bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    heatmap_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    return render_template('insights.html', genre_sales=genre_sales, platform_sales=platform_sales, plot_data=plot_data, heatmap_data=heatmap_data)



 


if __name__ == '__main__':
    app.run(debug=True)
