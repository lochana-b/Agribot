import os
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import requests
from gtts import gTTS
import random
import string

# Load API keys from .env file
load_dotenv()
hugging_face_api_key = os.getenv('hugging_face_api_key')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'webm'}

# Ensure the uploads and audio folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/audio', exist_ok=True)

def get_answer_mistral(question):
    """Fetches a response from Mistral-7B-Instruct via Hugging Face API."""
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    headers = {"Authorization": f"Bearer {hugging_face_api_key}"}
    
    data = {"inputs": question}
    response = requests.post(API_URL, headers=headers, json=data)
    
    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"]
        return "Error: Unexpected API response format."
    else:
        return f"Error: {response.json()}"


def text_to_audio(text, filename):
    """Converts text to speech and saves as an MP3 file."""
    output_dir = 'static/audio/'
    
    # Convert text to speech and save the file
    tts = gTTS(text)
    tts.save(os.path.join(output_dir, f"{filename}.mp3"))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    """Handles user input (text or audio) and returns AI-generated response with speech output."""
    if 'audio' in request.files:
        audio = request.files['audio']
        if audio and allowed_file(audio.filename):
            filename = secure_filename(audio.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            audio.save(filepath)
            transcription = process_audio(filepath)
            return jsonify({'text': transcription})

    text = request.form.get('text')
    if text:
        response = process_text(text)
        return jsonify({
            'text': response['text'],
            'voice': url_for('static', filename=f'audio/{response["voice"]}')
        })

    return jsonify({'text': 'Invalid request'})


def allowed_file(filename):
    """Checks if the uploaded file is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def process_audio(filepath):
    """Processes audio by sending it to Hugging Face's wav2vec2 model for transcription."""
    API_URL = "https://api-inference.huggingface.co/models/jonatasgrosman/wav2vec2-large-xlsr-53-english"
    headers = {"Authorization": f"Bearer {hugging_face_api_key}"}

    with open(filepath, "rb") as f:
        data = f.read()

    response = requests.post(API_URL, headers=headers, data=data)
    
    if response.status_code != 200:
        print("❌ API Error:", response.json())  # Debugging error
        return "Error processing audio"
    
    data = response.json()
    return data.get('text', "No transcription found")


def process_text(text):
    """Processes text by fetching a response from Mistral-7B-Instruct and converting it to audio."""
    return_text = get_answer_mistral(text)
    
    # Generate a random filename for the audio
    res = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    text_to_audio(return_text, res)
    
    return {"text": return_text, "voice": f"{res}.mp3"}


if __name__ == '__main__':
    app.run(debug=True)
