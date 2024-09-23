import torch
try:
    import torch
except ImportError:
    print("torch is not installed. Please install it using: pip install torch")
from PIL import Image
import requests
import base64
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define dangerous animals
dangerous_animals = [
    "lion", "tiger", "bear", "shark", "crocodile", "alligator", "snake",
    "scorpion", "jellyfish", "hippo", "rhino", "wolf", "leopard", "jaguar"
]

@app.route('/analyze', methods=['POST'])
def classify_image():
    # Get the image URL or base64 data from the request
    image_data = request.json.get('image_url') or request.json.get('image_data')
    try:
        if not image_data:
            raise KeyError('Missing image_url or image_data parameter')
    except KeyError as e:
        return jsonify({'error': str(e)}), 400

    try:
        # Check if the input is a URL or base64 data
        if image_data.startswith('http'):
            response = requests.get(image_data)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
        elif image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
            image = Image.open(BytesIO(base64.b64decode(image_data)))
        else:
            raise ValueError('Invalid image data format')

        # Prepare the text inputs
        text_inputs = ["a photo of a " + animal for animal in dangerous_animals]
        text_inputs.append("a photo of a safe animal")

        # Process the inputs
        inputs = processor(text=text_inputs, images=image, return_tensors="pt", padding=True)

        # Get the model outputs
        outputs = model(**inputs)

        # Calculate the probabilities
        probs = outputs.logits_per_image.softmax(dim=1)

        # Get the highest probability and its index
        max_prob, max_index = torch.max(probs, dim=1)

        # Determine if the animal is dangerous
        is_dangerous = max_index.item() < len(dangerous_animals)

        # Prepare the response
        response = {
            "is_dangerous": is_dangerous,
            "confidence": max_prob.item(),
            "animal": dangerous_animals[max_index.item()] if is_dangerous else "safe animal"
        }

        return jsonify(response)

    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Error fetching image: {str(e)}'}), 400
    except (ValueError, IOError) as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
