"""
Vietnamese Speech-to-Text Web Application
Sử dụng Flask với 3 mô hình: Wav2Vec2, PhoWhisper, OpenAI Whisper
"""

import os
import uuid
import torch
import librosa
import tempfile
import soundfile as sf
import numpy as np
import traceback
from flask import Flask, render_template, request, jsonify
from transformers import (
    Wav2Vec2ForCTC, 
    Wav2Vec2Processor,
    WhisperForConditionalGeneration,
    WhisperProcessor
)
from peft import PeftModel

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Global model storage
models = {}
processors = {}

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def load_wav2vec2():
    """Load Wav2Vec2 model"""
    if 'wav2vec2' not in models:
        print("Loading Wav2Vec2 model...")
        model_name = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
        processors['wav2vec2'] = Wav2Vec2Processor.from_pretrained(model_name)
        models['wav2vec2'] = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
        models['wav2vec2'].eval()
        print("Wav2Vec2 loaded successfully!")
    return models['wav2vec2'], processors['wav2vec2']

def load_phowhisper():
    """Load PhoWhisper model"""
    if 'phowhisper' not in models:
        print("Loading PhoWhisper model...")
        model_name = "vinai/PhoWhisper-small"
        processors['phowhisper'] = WhisperProcessor.from_pretrained(model_name)
        models['phowhisper'] = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        models['phowhisper'].eval()
        print("PhoWhisper loaded successfully!")
    return models['phowhisper'], processors['phowhisper']

def load_whisper():
    """Load OpenAI Whisper model"""
    if 'whisper' not in models:
        print("Loading OpenAI Whisper model...")
        model_name = "openai/whisper-small"
        processors['whisper'] = WhisperProcessor.from_pretrained(model_name)
        models['whisper'] = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        models['whisper'].eval()
        print("OpenAI Whisper loaded successfully!")
    return models['whisper'], processors['whisper']

def transcribe_wav2vec2(audio_path):
    """Transcribe using Wav2Vec2"""
    model, processor = load_wav2vec2()
    
    # Load and resample audio
    speech, sr = librosa.load(audio_path, sr=16000)
    
    # Process input
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Inference
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # Decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    
    return transcription

def transcribe_phowhisper(audio_path):
    """Transcribe using PhoWhisper"""
    model, processor = load_phowhisper()
    
    # Load and resample audio
    speech, sr = librosa.load(audio_path, sr=16000)
    
    # Process input
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Inference
    with torch.no_grad():
        generated_ids = model.generate(
            inputs["input_features"],
            max_length=225,
            language="vi",
            task="transcribe"
        )
    
    # Decode
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return transcription

def transcribe_whisper(audio_path):
    """Transcribe using OpenAI Whisper"""
    model, processor = load_whisper()
    
    # Load and resample audio
    speech, sr = librosa.load(audio_path, sr=16000)
    
    # Process input
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Inference
    with torch.no_grad():
        generated_ids = model.generate(
            inputs["input_features"],
            max_length=225,
            language="vi",
            task="transcribe"
        )
    
    # Decode
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return transcription

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Handle transcription request"""
    try:
        # Get model choice
        model_choice = request.form.get('model', 'phowhisper')
        
        # Check if file was uploaded
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Generate unique filename with original extension
        original_ext = os.path.splitext(audio_file.filename)[1] or '.webm'
        unique_id = str(uuid.uuid4())[:8]
        temp_input = os.path.join(app.config['UPLOAD_FOLDER'], f'input_{unique_id}{original_ext}')
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f'audio_{unique_id}.wav')
        
        # Save uploaded file
        audio_file.save(temp_input)
        
        # Convert to WAV using librosa (handles webm, ogg, mp3, etc.)
        try:
            audio, sr = librosa.load(temp_input, sr=16000)
            sf.write(temp_path, audio, 16000)
        except Exception as e:
            # If librosa fails, try using the original file
            print(f"Audio conversion warning: {e}")
            temp_path = temp_input
        
        # Transcribe based on model choice
        if model_choice == 'wav2vec2':
            transcription = transcribe_wav2vec2(temp_path)
        elif model_choice == 'phowhisper':
            transcription = transcribe_phowhisper(temp_path)
        elif model_choice == 'whisper':
            transcription = transcribe_whisper(temp_path)
        else:
            return jsonify({'error': 'Invalid model choice'}), 400
        
        # Clean up temp files
        for f in [temp_path, temp_input]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass
        
        return jsonify({
            'success': True,
            'transcription': transcription,
            'model': model_choice
        })
        
    except Exception as e:
        print(f"ERROR in /transcribe: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'device': device,
        'models_loaded': list(models.keys())
    })

if __name__ == '__main__':
    print("=" * 50)
    print("Vietnamese Speech-to-Text Application")
    print("=" * 50)
    print(f"Device: {device}")
    print("Available models: Wav2Vec2, PhoWhisper, OpenAI Whisper")
    print("=" * 50)
    
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True)
