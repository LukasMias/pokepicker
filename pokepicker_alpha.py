import streamlit as st
import pandas as pd
import numpy as np
from skimage import io
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from PIL import Image
import os

# Definiere die Funktionen get_main_colors und get_matches
st.title('Pokeball-Picker')
def get_main_colors(image, n_colors=3):
    # Entferne den Alpha-Kanal und flache das Bild ab
    pixels = image[:,:,:3].reshape(-1, 3)
    # Entferne transparente/weiße Pixel
    pixels = pixels[~np.all(pixels >= [240, 240, 240], axis=1)]
    
    # Verwende KMeans, um die n_colors Hauptfarben des Bildes zu finden
    kmeans = KMeans(n_clusters=n_colors,random_state=80)
    kmeans.fit(pixels)
    return kmeans.cluster_centers_

def match_colors(pokemon_colors, ball_colors):
    # Berechne die Distanz zwischen den Hauptfarben des Pokémons und der Bälle
    distances = pairwise_distances(pokemon_colors, ball_colors)
    # Finde die Ballfarbe mit der geringsten Gesamtdistanz zu den Pokémonfarben
    index = np.sum(distances, axis=0).argmin()
    return index


# Lade den DataFrame
df = pd.read_feather('pokemondf.feather')

# Auswahl des Pokémon
pokemon_choice = st.selectbox("Wähle ein Pokémon:", df.dropna()['Deutscher Name'].unique())
pokemon_id = df.loc[df['Deutscher Name'] == pokemon_choice, 'ID'].astype(int).values[0]

# Pfade für Sprites und Bälle
balls_folder = 'balls' # Pfad zum Balls-Ordner
sprites_folder = 'sprites' # Pfad zum Sprites-Ordner

# Shiny-Auswahl und Sprite-Pfade
is_shiny = st.checkbox("Shiny-Version auswählen")
normal_sprite_path = f'{sprites_folder}/{pokemon_id}.png'
shiny_sprite_path = f'{sprites_folder}/{pokemon_id}_shiny.png'
chosen_sprite_path = shiny_sprite_path if is_shiny else normal_sprite_path
sprite_image = io.imread(chosen_sprite_path)
pokemon_colors = get_main_colors(sprite_image)

# Zeige das ausgewählte Sprite

balls_images = {name: io.imread(os.path.join('balls', name)) for name in os.listdir('balls')} 
balls_colors = {name: get_main_colors(img) for name, img in balls_images.items()}

# Berechne die Übereinstimmungszahlen
matches = {}
for name, colors in balls_colors.items():
    # Berechne die Distanz zwischen den Hauptfarben des Pokémons und der Bälle
    distances = pairwise_distances(pokemon_colors, colors)
    # Summiere die Distanzen für jede Ballfarbe
    match_score = np.sum(distances, axis=0).min()
    matches[name] = match_score

# Normiere die Übereinstimmungszahlen zwischen 0 und 1
max_score = max(matches.values())
min_score = min(matches.values())
normalized_matches = {name: (max_score - score) / (max_score - min_score) for name, score in matches.items()}

# Sortiere die Bälle nach ihrer Übereinstimmungszahl und wähle die besten drei aus
best_three_balls = sorted(normalized_matches.items(), key=lambda item: item[1], reverse=True)[:3]

# Gib die besten drei Bälle aus
col1, col2 = st.columns(2)
with col1:
    st.image(chosen_sprite_path,use_column_width=True)
with col2:
    for ball, score in best_three_balls:
        st.image(f'balls/{ball}')
        st.write(f"Ball: {ball}, Übereinstimmung: {score:.3f}")

