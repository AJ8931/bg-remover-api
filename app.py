from flask import Flask, request, jsonify
from transformers import pipeline
from PIL import Image
import requests
import io
import base64

app = Flask(__name__)

# Initialize the image segmentation pipeline
pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)

@app.route("/remove-bg", methods=["POST"])
def remove_bg():
    if "image_url" not in request.json:
        return jsonify({"error": "Image URL is required"}), 400

    image_url = request.json["image_url"]

    # Download the image
    response = requests.get(image_url)
    image = Image.open(io.BytesIO(response.content))

    # Apply the image segmentation model
    pillow_image = pipe(image)

    # Convert to bytes to return as a base64 string
    img_byte_arr = io.BytesIO()
    pillow_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    return jsonify({"message": "Background removed successfully", "image": img_base64})

if __name__ == "__main__":
    app.run(debug=True)
