from flask import Flask, request, render_template
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)

model = tf.keras.models.load_model('path_to_your_model.h5')

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def predict_image(image_path):
    processed_image = load_and_preprocess_image(image_path)
    predictions = model.predict(processed_image)
    return predictions

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        file.save('uploaded_image.jpg')
        predictions = predict_image('uploaded_image.jpg')
        return render_template('result.html', predictions=predictions)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
