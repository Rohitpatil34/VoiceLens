# Importing required libraries
from flask import Flask, render_template, request, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import torch
import uuid
import os
from gtts import gTTS
from googletrans import Translator
import json


# Step 1: Flickr8k Dataset Integration Functions

def load_flickr8k_captions(flickr_captions_file):
    """
    Load and parse captions from Flickr8k dataset file.
    
    Args:
        flickr_captions_file (str): Path to the Flickr8k captions file
        
    Returns:
        dict: Dictionary mapping image IDs to their captions
    """
    captions_dict = {}
    
    try:
        with open(flickr_captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Clean and validate each line
                line = line.strip()
                
                if not line or "|" not in line:
                    continue
                    
                # Split image name and caption
                parts = line.split('\t')
                
                if len(parts) < 2:
                    continue
                    
                img, caption = parts[0], parts[1]
                
                # Extract image ID (removes #0, #1 etc.)
                img_id = img.split('#')[0]
                
                # Initialize list if needed and append caption
                if img_id not in captions_dict:
                    captions_dict[img_id] = []
                
                captions_dict[img_id].append(caption)
                
    except Exception as e:
        print(f"Error loading Flickr8k captions: {str(e)}")
        
    return captions_dict


# Load Flickr8k captions if dataset exists
FLICKR8K_CAPTIONS_PATH = 'D:\project 2 (blip)\archive'

if os.path.exists(FLICKR8K_CAPTIONS_PATH):
    flickr_captions = load_flickr8k_captions(FLICKR8K_CAPTIONS_PATH)
else:
    flickr_captions = {}
    print("Flickr8k captions file not found. Proceeding without dataset integration.")


# Step 2: Initialize Flask Application and BLIP Model

# Create Flask application instance
app = Flask(__name__)

# Initialize BLIP processor and model
try:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
except Exception as e:
    print(f"Error loading BLIP model: {str(e)}")
    raise

# Initialize translator
translator = Translator()


# Flask Routes

@app.route('/')
def index():
    """
    Serve the main index page.
    """
    return render_template('index.html')


@app.route('/generate_caption', methods=['POST'])
def generate_caption():
    """
    Handle caption generation requests.
    
    Expects JSON with:
    - image_url: URL of the image to caption
    - lang: (optional) target language for translation
    
    Returns JSON with:
    - caption: original English caption
    - translated_caption: translated caption (if requested)
    - audio_url: path to generated audio file
    """
    # Get request data
    data = request.get_json()
    
    # Validate required fields
    if not data or 'image_url' not in data:
        return jsonify({'error': 'Missing image_url parameter'}), 400
        
    image_url = data['image_url']
    lang = data.get('lang', 'en')
    
    # Download and process image
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        
        image = Image.open(response.raw).convert('RGB')
        
    except Exception as e:
        error_msg = f"Could not load image: {str(e)}"
        return jsonify({'error': error_msg}), 400
        
    # Generate caption using BLIP model
    try:
        inputs = processor(image, return_tensors="pt")
        
        with torch.no_grad():
            out = model.generate(**inputs)
            
        caption = processor.decode(out[0], skip_special_tokens=True)
        
    except Exception as e:
        error_msg = f"Caption generation failed: {str(e)}"
        return jsonify({'error': error_msg}), 500
        
    # Handle translation if requested
    translated_caption = caption
    
    if lang != 'en':
        try:
            translation = translator.translate(caption, dest=lang)
            translated_caption = translation.text
            
        except Exception as e:
            error_msg = f"Translation failed: {str(e)}"
            return jsonify({'error': error_msg}), 500
            
    # Generate audio file
    try:
        audio = gTTS(text=translated_caption, lang=lang)
        
        # Create unique filename
        audio_filename = f"static/audio_{uuid.uuid4()}.mp3"
        
        # Ensure static directory exists
        os.makedirs("static", exist_ok=True)
        
        # Save audio file
        audio.save(audio_filename)
        
    except Exception as e:
        error_msg = f"Audio generation failed: {str(e)}"
        return jsonify({'error': error_msg}), 500
        
    # Prepare response
    response_data = {
        'caption': caption,
        'translated_caption': translated_caption,
        'audio_url': f'/{audio_filename}'
    }
    
    return jsonify(response_data)


# Main Application Entry Point

if __name__ == '__main__':
    # Create static directory if it doesn't exist
    if not os.path.exists("static"):
        os.makedirs("static")
        
    # Run Flask application
    app.run(debug=True, host='0.0.0.0', port=5000)