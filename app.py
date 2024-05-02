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
st.set_page_config(page_title='PokéballPicker',page_icon='cover.jpg',layout='centered')
balldict = {
    'Freundesball': 'Freundesball.png',
    'Wiederball': 'Wiederball.png',
    'Tauchball': 'Tauchball.png',
    'Nestball': 'Nestball.png',
    'Timerball': 'Timerball.png',
    'Hyperball': 'Hyperball.png',
    'Flottball': 'Flottball.png',
    'Heilball': 'Heilball.png',
    'Levelball': 'Levelball.png',
    'Traumball': 'Traumball.png',
    'Sympaball': 'Sympaball.png',
    'Superball': 'Superball.png',
    'Luxusball': 'Luxusball.png',
    'Mondball': 'Mondball.png',
    'Koederball': 'Koederball.png',
    'Schwerball': 'Schwerball.png',
    'Premierball': 'Premierball.png',
    'Pokeball': 'Pokeball.png',
    'Finsterball': 'Finsterball.png',
    'Safariball': 'Safariball.png',
    'Netzball': 'Netzball.png',
    'Meisterball': 'Meisterball.png',
    'Jubelball': 'Jubelball.png',
    'Rätselball': 'Raetselball.png',
    'Turnierball': 'Turnierball.png',
    'Ultraball': 'Ultraball.png',
    'Turboball':'Turboball.png'
}


# Funktion zum Laden aller Pokéball-Bilder beim Start der App
if 'ball_images' not in st.session_state:
    st.session_state['ball_images'] = {}
    for key, value in balldict.items():
        img = Image.open(value).convert('RGBA')
        st.session_state['ball_images'][key] = img




def load_image(image_path):
    response = requests.get(image_path)
    response.raise_for_status()  # Stellt sicher, dass die Anfrage erfolgreich war
    img = Image.open(BytesIO(response.content)).convert('RGBA')
    return img

def get_main_colors(img, n_colors):   
    data = np.array([pixel[:3] for pixel in img.getdata() if pixel[3] != 0], np.uint8)
    kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(data)
    centers = kmeans.cluster_centers_
    return centers

def match_colors(pokemon_colors, ball_colors):
    distances = pairwise_distances(pokemon_colors, ball_colors)
    index = np.sum(distances, axis=0).argmin()
    return index


if 'data' not in st.session_state:
    st.session_state['data'] = pd.read_excel('pokemondf.xlsx')

with open('color_data.json', 'r') as file:
    balls_colors = json.load(file)

if 'remover' not in st.session_state:
    st.session_state['remover'] = Remover(mode='base-nightly')
df = st.session_state['data'].copy()

st.title('Pokeball-Picker')
st.sidebar.write('In dieser App kannst du ein Pokémon auswählen und erhältst anschließend Vorschläge für die zu dem Sprite passenden Bälle! Solltest du mit der Darstellung des Sprites unzufrieden sein, kannst du im unteren Teil der App auch ein eigenes Bild hochladen und dir entsprechende Bälle empfehlen lassen. Die Basis für die Empfehlungen bilden die "Hauptfarben" des Pokémon, die mit den Hauptfarben der Bälle verglichen werden.')
clusterzahl = st.sidebar.slider('Wie viele Hauptfarben sollen berücksichtigt werden?', min_value=1, max_value=5, step=1, value=2)

pokemon_choice = st.selectbox("Wähle ein Pokémon:", df['germanname'].unique())
row = df[df['germanname'] == pokemon_choice].iloc[0]
is_shiny = st.checkbox("Shiny-Version auswählen")
normal_sprite_path = row.sprite
shiny_sprite_path = row.shiny_sprite
chosen_sprite_path = shiny_sprite_path if is_shiny else normal_sprite_path
pokemon_sprite = load_image(chosen_sprite_path)
pokemon_colors = get_main_colors(pokemon_sprite, n_colors=clusterzahl)

matches = {}
for name, colors in balls_colors.items():
    distances = pairwise_distances(pokemon_colors, np.array(colors))
    match_score = np.sum(distances, axis=0).min()
    matches[name] = match_score

# Sortiere die Bälle nach ihrem Match-Score und extrahiere die Namen der besten drei Bälle
best_three_balls = [name for name, score in sorted(matches.items(), key=lambda item: item[1])[:3]]


col1, col2, col3 = st.columns([0.5, 0.25, 0.25])
with col1:
    st.subheader('Sprite')
    st.image(chosen_sprite_path, use_column_width=True)

with col2:
    random_choice = random.randint(0, 5)
    if random_choice == 2:
        st.subheader('Best Ball \n (Dominant Color)')
        st.image(st.session_state['ball_images']['Flottball'])
        st.image(st.session_state['ball_images']['Flottball'])  
        st.image(st.session_state['ball_images']['Flottball'])  
    else:
        st.subheader('Best Balls \n (Dominant Color)')
        for ball in best_three_balls:
            if ball in st.session_state['ball_images']:
                st.image(st.session_state['ball_images'][ball], use_column_width=False)


# Auswahl der richtigen Spalten basierend auf 'is_shiny'
if is_shiny:
    ball_columns = ['shiny_ball_1', 'shiny_ball_2', 'shiny_ball_3']
else:
    ball_columns = ['ball_1', 'ball_2', 'ball_3']


with col3:
    st.subheader('Best Balls \n (Human choice)')
    for col in ball_columns:
        if pd.notna(row[col]):
            st.image(st.session_state['ball_images'][row[col]], use_column_width=False)

img_file_buffer = st.file_uploader("Eigenes Bild hochladen", type=["png", "jpg", "jpeg"])
if img_file_buffer is not None:
    image = Image.open(img_file_buffer)

    # Originalabmessungen auslesen
    original_width, original_height = image.size

    # Bestimmen, ob die Höhe oder die Breite größer ist und entsprechend skalieren
    if original_width > original_height:
        # Breite ist größer, also Breite auf 400 setzen
        scale_factor = 400 / original_width
        new_width = 400
        new_height = int(original_height * scale_factor)
    else:
        # Höhe ist größer, also Höhe auf 400 setzen
        scale_factor = 400 / original_height
        new_height = 400
        new_width = int(original_width * scale_factor)

    # Bild skalieren
    image = image.resize((new_width, new_height), Image.LANCZOS)



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
        upload_colors = get_main_colors(out, n_colors=clusterzahl)
        matches = {}
        for name, colors in balls_colors.items():
            distances = pairwise_distances(upload_colors, np.array(colors))
            match_score = np.sum(distances, axis=0).min()
            matches[name] = match_score
        best_three_balls = [name for name, score in sorted(matches.items(), key=lambda item: item[1])[:3]]

        st.subheader('Best Balls \n (Dominant Color)')
        for ball in best_three_balls:
            if ball in st.session_state['ball_images']:
                st.image(st.session_state['ball_images'][ball], use_column_width=False)

footer = """
<style>
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
st.markdown(footer, unsafe_allow_html=True)
