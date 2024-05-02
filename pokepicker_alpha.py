import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from PIL import Image
import os
import random
from sklearn.cluster import MiniBatchKMeans
import json
import requests
from io import BytesIO
from transparent_background import Remover

balldict = {
    'Freundesball': 'https://i.imgur.com/VQL4lPa.png',
    'Wiederball': 'https://i.imgur.com/RqSOGo6.png',
    'Tauchball': 'https://i.imgur.com/XNzvO0H.png',
    'Nestball': 'https://i.imgur.com/XIu6jDl.png',
    'Timerball': 'https://i.imgur.com/JOd7Boe.png',
    'Hyperball': 'https://i.imgur.com/beHgI95.png',
    'Turboball': 'https://i.imgur.com/lKAfx6k.png',
    'Heilball': 'https://i.imgur.com/rAI9Gyi.png',
    'Levelball': 'https://i.imgur.com/rJhTvoD.png',
    'Turboball': 'https://i.imgur.com/zpybKax.png',
    'Traumball': 'https://i.imgur.com/XI18XGu.png',
    'Sympaball': 'https://i.imgur.com/a9HBljD.png',
    'Superball': 'https://i.imgur.com/SF5CZ6U.png',
    'Luxusball': 'https://i.imgur.com/EkqYYOB.png',
    'Mondball': 'https://i.imgur.com/JDgAgAO.png',
    'Koederball': 'https://i.imgur.com/ef1o2lD.png',
    'Schwerball': 'https://i.imgur.com/NkILW1J.png',
    'Premierball': 'https://i.imgur.com/0ArMZjJ.png',
    'Pokeball': 'https://i.imgur.com/bxys1sd.png',
    'Finsterball': 'https://i.imgur.com/N2Wqr43.png',
    'Safariball': 'https://i.imgur.com/lpOVjtL.png',
    'Netzball': 'https://i.imgur.com/XzidFHI.png',
    'Meisterball': 'https://i.imgur.com/nNTmuH8.png',
    'Jubelball': 'https://i.imgur.com/aeqHLEh.png',
    'Rätselball': 'https://i.imgur.com/eru43o1.png',
    'Turnierball': 'https://i.imgur.com/GyNf8BH.png',
    'Ultraball' : 'https://imgur.com/IIajvuS'
}





# Definiere die Funktionen get_main_colors und get_matches

@st.cache_data
def load_image(image_path):
    response = requests.get(image_path)
    response.raise_for_status()  # Stellt sicher, dass die Anfrage erfolgreich war
    # Bytes in ein PIL-Image umwandeln
    img = Image.open(BytesIO(response.content)).convert('RGBA')
    return img

def get_main_colors(img, n_colors):   
    # Convert cropped image to numpy array and filter non-transparent pixels
    data = np.array([pixel[:3] for pixel in img.getdata() if pixel[3] != 0], np.uint8)

    # Clustering
    kmeans = MiniBatchKMeans(n_clusters=n_colors,random_state=42)
    kmeans.fit(data)
    centers = kmeans.cluster_centers_
    return centers

def match_colors(pokemon_colors, ball_colors):
    # Berechne die Distanz zwischen den Hauptfarben des Pokémons und der Bälle
    distances = pairwise_distances(pokemon_colors, ball_colors)
    # Finde die Ballfarbe mit der geringsten Gesamtdistanz zu den Pokémonfarben
    index = np.sum(distances, axis=0).argmin()
    return index


# Lade den DataFrame
if 'data' not in st.session_state:
    st.session_state['data'] = pd.read_excel('pokemondf.xlsx')

with open('color_data.json', 'r') as file:
    balls_colors = json.load(file)

if 'remover' not in st.session_state:
    st.session_state['remover'] = Remover(mode='base-nightly')
df = st.session_state['data'].copy()

st.title('Pokeball-Picker')
st.sidebar.write('In dieser App kannst du ein Pokémon auswählen und erhältst anschließend Vorschläge für die zu dem Sprite passenden Bälle! Solltest du mit der Darstellung des Sprites unzufrieden sein, kannst du im unteren Teil der App auch ein eigenes Bild hochladen und dir entsprechende Bälle empfehlen lassen. Die Basis für die Empfehlungen bilden die "Hauptfarben" des Pokémon, die mit den Hauptfarben der Bälle verglichen werden.')
clusterzahl = st.sidebar.slider('Wie viele Hauptfarben sollen berücksichtigt werden?',min_value=1,max_value=5,step=1,value=2)
# Auswahl des Pokémon
pokemon_choice = st.selectbox("Wähle ein Pokémon:", df['germanname'].unique())

# Pfade für Sprites und Bälle
balls_folder = 'balls' # Pfad zum Balls-Ordner

row = row = df[df['germanname'] == pokemon_choice].iloc[0]

# Shiny-Auswahl und Sprite-Pfade
is_shiny = st.checkbox("Shiny-Version auswählen")
normal_sprite_path = row.sprite
shiny_sprite_path = row.shiny_sprite
chosen_sprite_path = shiny_sprite_path if is_shiny else normal_sprite_path
pokemon_sprite = load_image(chosen_sprite_path)
pokemon_colors = get_main_colors(pokemon_sprite,n_colors=clusterzahl)

# Berechne die Übereinstimmungszahlen
matches = {}
for name, colors in balls_colors.items():
    # Berechne die Distanz zwischen den Hauptfarben des Pokémons und der Bälle
    distances = pairwise_distances(pokemon_colors, np.array(colors))
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
col1, col2, col3 = st.columns([0.5,0.25,0.25])
with col1:
    st.subheader('Sprite')
    st.image(chosen_sprite_path, use_column_width=True)

# Zufällige Auswahl
random_choice = random.randint(0,5)
if random_choice == 2:
    with col2:
        st.subheader('Best Ball \n (Dominant Color)')
        for ball in best_three_balls:
            st.image(os.path.join('balls', 'flottball.png'))  # Angenommen, 'flottball.png' ist ein Platzhalter
else:
    with col2:
        st.subheader('Best Balls \n (Dominant Color)')
        for ball, score in best_three_balls:
            st.image(os.path.join('balls', ball))



# Umkehren des balldict, um von URLs auf Namen zu mappen
url_to_name = {v: k for k, v in balldict.items()}

# Auswahl der richtigen Spalten basierend auf 'is_shiny'
if is_shiny:
    ball_columns = ['shiny_ball_1', 'shiny_ball_2', 'shiny_ball_3']
else:
    ball_columns = ['ball_1', 'ball_2', 'ball_3']

# Extrahiere die URLs direkt aus den DataFrame-Spalten und konvertiere sie in lokale Pfade
ball_filenames = []
for col in ball_columns:
    if row[col] is not None:
        # Ermittle den Namen des Balls aus der URL
        ball_name = url_to_name.get(row[col])
        if ball_name:
            # Generiere den lokalen Pfadnamen basierend auf dem Ballnamen
            local_path = os.path.join('balls', f'{ball_name}.png')
            ball_filenames.append(local_path)


with col3:
    st.subheader('Best Balls \n (Human choice)')
    for filename in ball_filenames:
        if filename:  # Überprüfen, ob der Pfad nicht None ist
            st.image(filename, use_column_width=False)





img_file_buffer = st.file_uploader("Eigenes Bild hochladen", type=["png", "jpg", "jpeg"])

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    # Überprüfe, ob das Bild bereits im RGBA-Format ist
    if image.mode != 'RGBA' and 'processed_image' not in st.session_state:
        st.warning('Achtung, die Farben des Hintergrunds deines Pokemons, gehen mit in die Kalkulation ein.')
        keinetransparenz = True
        out = image.convert('RGBA')
    else:
        if 'processed_image' in st.session_state:
            out = st.session_state['processed_image']
        keinetransparenz = False


    col1, col3 = st.columns([0.75,0.25])
    with col1:
        st.image(
            out,
            use_column_width=True,
        )
        if keinetransparenz:
            st.write('Mit folgendem Knopf kannst du den Hintergrund des Pokemons auf deinem Bild entfernen lassen. Beachte jedoch, dass der Algorithmus je nach Größe des Bildes einige Zeit in Anspruch nehmen kann.')
            if st.button('Hintergrund entfernen!'):
                with st.spinner('Hintergrund wird übermalt...'):
                    st.session_state['processed_image'] = st.session_state['remover'].process(image)
                    st.rerun()
        if 'processed_image' in st.session_state:
            if st.button('Bild zurücksetzen!'):
                del st.session_state['processed_image']
                st.rerun()

    with col3:
        # Berechne die Übereinstimmungszahlen
        upload_colors = get_main_colors(out,n_colors=clusterzahl)
        matches = {}
        for name, colors in balls_colors.items():
            # Berechne die Distanz zwischen den Hauptfarben des Pokémons und der Bälle
            distances = pairwise_distances(upload_colors, np.array(colors))
            # Summiere die Distanzen für jede Ballfarbe
            match_score = np.sum(distances, axis=0).min()
            matches[name] = match_score

        # Normiere die Übereinstimmungszahlen zwischen 0 und 1
        max_score = max(matches.values())
        min_score = min(matches.values())
        normalized_matches = {name: (max_score - score) / (max_score - min_score) for name, score in matches.items()}

        # Sortiere die Bälle nach ihrer Übereinstimmungszahl und wähle die besten drei aus
        best_three_balls = sorted(normalized_matches.items(), key=lambda item: item[1], reverse=True)[:3]

        st.subheader('Best Balls \n (Dominant Color)')
        for ball, score in best_three_balls:
            st.image(os.path.join('balls', ball))






    # # default setting - transparent background





#############Fußnote einfügen
footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Human Choice is based on reddit user OracleLink <a style='display: block; text-align: center;' href="https://docs.google.com/spreadsheets/d/1bvIx7Q2Lxp7efHRrUh48WkuwirNlKardwSHVz_R8kA0/edit#gid=1553039354" target="_blank">And his spreadsheet</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)