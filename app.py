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

 
def load_flickr8k_captions(flickr_captions_file):
    """Load and parse captions from Flickr8k dataset ."""
    captions_dict = {}
    with open(flickr_captions_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            img, caption = line.split('\t')
            img_id = img.split('#')[0]
            captions_dict.setdefault(img_id, []).append(caption)
    return captions_dict

# Load  captions 
FLICKR8K_CAPTIONS_PATH = 'D:\project 2 (blip)\archive'  # Ensure this file is in your working directory
if os.path.exists(FLICKR8K_CAPTIONS_PATH):
    flickr_captions = load_flickr8k_captions(FLICKR8K_CAPTIONS_PATH)
else:
    flickr_captions = {}

#  Step 2: BLIP + Flask App for Actual Caption Generation 
app = Flask(__name__)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
translator = Translator()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_caption', methods=['POST'])
def generate_caption():
    data = request.get_json()
    image_url = data['image_url']
    lang = data.get('lang', 'en')

    try:
        image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
    except Exception as e:
        return jsonify({'error': f"Could not load image: {str(e)}"}), 400

    # Generate caption using BLIP
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    # Optional translation
    if lang != 'en':
        try:
            translated_caption = translator.translate(caption, dest=lang).text
        except Exception as e:
            return jsonify({'error': f"Translation failed: {str(e)}"}), 500
    else:
        translated_caption = caption

    # Generate speech
    audio = gTTS(text=translated_caption, lang=lang)
    audio_filename = f"static/audio_{uuid.uuid4()}.mp3"
    audio.save(audio_filename)

    return jsonify({
        'caption': caption,
        'translated_caption': translated_caption,
        'audio_url': '/' + audio_filename
    })

if __name__ == '__main__':
    os.makedirs("static", exist_ok=True)
    app.run(debug=True)
