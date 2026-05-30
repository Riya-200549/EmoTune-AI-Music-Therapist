from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import base64

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024

model = load_model("model/emotion_model.h5")

emotions = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Neutral',
    'Sad',
    'Surprise'
]

song_dict = {
    "Angry": "calm.mp3",
    "Disgust": "peace.mp3",
    "Fear": "calm.mp3",
    "Happy": "happy.mp3",
    "Neutral": "peace.mp3",
    "Sad": "motivation.mp3",
    "Surprise": "happy.mp3"
}

@app.route('/songs/<filename>')
def songs(filename):

    song_folder = os.path.join(
        app.root_path,
        'static',
        'songs'
    )

    return send_from_directory(
        song_folder,
        filename
    )

@app.route('/', methods=['GET', 'POST'])
def home():

    prediction = ""
    song = ""
    image_path = ""

    if request.method == 'POST':

        filepath = ""

        if 'image' in request.files:

            image = request.files['image']

            if image and image.filename != "":

                os.makedirs("static", exist_ok=True)

                filepath = os.path.join(
                    "static",
                    image.filename
                )

                image.save(filepath)

                image_path = "/" + filepath.replace("\\", "/")

        captured_image = request.form.get("captured_image")

        if captured_image:

            try:

                image_data = captured_image.split(",")[1]

                image_bytes = base64.b64decode(image_data)

                filepath = os.path.join(
                    "static",
                    "captured.jpg"
                )

                with open(filepath, "wb") as f:
                    f.write(image_bytes)

                image_path = "/static/captured.jpg"

            except:
                pass

        if filepath != "":

            img = cv2.imread(
                filepath,
                cv2.IMREAD_GRAYSCALE
            )

            if img is not None:

                img = cv2.resize(
                    img,
                    (48, 48)
                )

                img = img / 255.0

                img = img.reshape(
                    1,
                    48,
                    48,
                    1
                )

                pred = model.predict(img)

                prediction = emotions[
                    np.argmax(pred)
                ]

                song = song_dict.get(
                    prediction,
                    "peace.mp3"
                )

            else:

                prediction = "Image Read Error"

    return render_template(
        "index.html",
        prediction=prediction,
        song=song,
        image_path=image_path
    )

if __name__ == '__main__':
    app.run(debug=True)
