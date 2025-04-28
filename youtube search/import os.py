import os
import sqlite3
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, redirect, render_template, send_from_directory, url_for
from openclip_module import OpenClip


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'mov', 'avi', 'mkv'}

openclip = OpenClip()




# ✅ Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# 🧱 Step 2: Initialize database
def init_db():
    conn = sqlite3.connect('videos.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            filename TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Call this function at app startup
init_db()

# 🏠 Home route
@app.route('/')
def index():
    return render_template('index.html')

# 📤 Upload video route
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        title = request.form['title']
        file = request.files['video']
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Step 1: Insert video into DB
            conn = sqlite3.connect('videos.db')
            c = conn.cursor()
            c.execute('INSERT INTO videos (title, filename) VALUES (?, ?)', (title, filename))
            video_id = c.lastrowid  # Get the new video ID
            conn.commit()
            conn.close()

            # Step 2: Extract and encode frames
            extract_and_encode_frames(filepath, video_id)

            return redirect(url_for('search'))
    return render_template('upload.html')

def extract_and_encode_frames(video_path, video_id):
    cap = cv2.VideoCapture(video_path)
    frame_features = []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_rate = max(1, frame_count // 5)  # Grab 5 frames total

    frame_idx = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        if frame_idx % sample_rate == 0:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            feature = openclip.encode_image(pil_img)
            frame_features.append(feature.cpu().numpy())
        frame_idx += 1

    cap.release()

    if frame_features:
        os.makedirs('frame_features', exist_ok=True)
        features_array = np.concatenate(frame_features, axis=0)
        np.save(f'frame_features/video_{video_id}.npy', features_array)

# 🎥 Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# 🔍 Search from database
@app.route('/search')
def search():
    query = request.args.get('q', '').lower().strip()
    query_embedding = openclip.encode_text(query).cpu().numpy()  # shape: (1, D)

    conn = sqlite3.connect('videos.db')
    c = conn.cursor()
    c.execute('SELECT id, title, filename FROM videos')
    all_videos = c.fetchall()
    conn.close()

    results = []

    for vid_id, title, filename in all_videos:
        feature_path = f'frame_features/video_{vid_id}.npy'
        if os.path.exists(feature_path):
            frame_feats = np.load(feature_path)  # shape (N, D)
            sims = np.dot(frame_feats, query_embedding.T).squeeze()  # (N,)
            max_sim = float(np.max(sims))  # max similarity score for this video

            if max_sim >= 0.3:  # filter out weak matches
                results.append((max_sim, {'title': title, 'filename': filename}))

    # Sort best matches to top
    sorted_results = sorted(results, key=lambda x: x[0], reverse=True)
    top_matches = [item[1] for item in sorted_results]

    return render_template('results.html', results=top_matches, query=query)


@app.route('/clear')
def clear_database():
    conn = sqlite3.connect('videos.db')
    c = conn.cursor()
    c.execute('DELETE FROM videos')  # Clear all rows
    conn.commit()
    conn.close()
    return "✅ All videos have been deleted from the database!"


# 🚀 Run the app
if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
