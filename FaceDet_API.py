import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
import io
from PIL import Image

app = Flask(__name__)


# Load the pre-trained Haar Cascade model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_faces(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image, faces




@app.route('/detect_faces', methods=['POST'])
def detect_faces_api():
    # Check if a file is uploaded
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    
    # Read the image file as a NumPy array
    img_array = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Invalid image file"}), 400

    # Detect faces in the image
    processed_image, faces = detect_faces(image)

    # Convert the processed image to PIL Image for easy response
    pil_img = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
    
    # Save the processed image to a byte buffer
    img_io = io.BytesIO()
    pil_img.save(img_io, 'PNG')
    img_io.seek(0)

    # Return the processed image as a response
    return send_file(img_io, mimetype='image/png')

@app.route('/')
def home():
    return "Fashion MNIST Model API - Use /predict to make predictions!"

if __name__ == '__main__':
    app.run(debug=True)
