import os
import sqlite3
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
import cv2
import torch
import numpy as np
from PIL import Image
from openclip_module import OpenClip  # <-- updated import

# Instantiate OpenCLIP
openclip_model = OpenClip()

# Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
EMBEDDING_FOLDER = 'embeddings'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EMBEDDING_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize the database
def init_db():
    with sqlite3.connect('videos.db') as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                filename TEXT
            )
        ''')
init_db()

# Function to extract frames and save embeddings
def extract_and_save_embeddings(video_path, video_id):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    frame_count = 0
    embeddings = []

    while success:
        if frame_count % 30 == 0:  # Every 30 frames (~1 second for 30fps video)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            image_features = openclip_model.encode_image(image_pil).cpu().numpy()
            embeddings.append(image_features)
        
        success, image = vidcap.read()
        frame_count += 1

    if embeddings:
        embeddings = np.vstack(embeddings)
        np.save(os.path.join(EMBEDDING_FOLDER, f'{video_id}.npy'), embeddings)

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Upload page
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        title = request.form['title']
        file = request.files['video']

        if file and file.filename.endswith(('.mp4', '.mov', '.avi', '.mkv')):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            with sqlite3.connect('videos.db') as conn:
                cur = conn.cursor()
                cur.execute('INSERT INTO videos (title, filename) VALUES (?, ?)', (title, filename))
                conn.commit()
                video_id = cur.lastrowid

            # Extract frames and generate embeddings
            extract_and_save_embeddings(filepath, video_id)

            return redirect(url_for('index'))

    return render_template('upload.html')

# Search page
@app.route('/search')
def search():
    query = request.args.get('q')
    if not query:
        return redirect(url_for('index'))

    # Encode the text query
    query_embedding = openclip_model.encode_text(query).cpu().numpy()

    results = []
    with sqlite3.connect('videos.db') as conn:
        cur = conn.cursor()
        cur.execute('SELECT id, title, filename FROM videos')
        videos = cur.fetchall()

    for video_id, title, filename in videos:
        embedding_path = os.path.join(EMBEDDING_FOLDER, f'{video_id}.npy')
        if os.path.exists(embedding_path):
            frame_embeddings = np.load(embedding_path)
            similarities = frame_embeddings @ query_embedding.T
            best_similarity = similarities.max()
            results.append({'id': video_id, 'title': title, 'filename': filename, 'similarity': best_similarity})

    # Sort videos by best similarity (higher is better)
    results.sort(key=lambda x: x['similarity'], reverse=True)

    # Keep only top 2 results
    results = results[:2]

    return render_template('results.html', results=results, query=query)


# Serve uploaded video files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Run app
if __name__ == '__main__':
    app.run(debug=True)
