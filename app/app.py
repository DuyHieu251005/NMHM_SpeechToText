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
import imageio_ffmpeg
from flask import Flask, render_template, request, jsonify

# Add FFmpeg to PATH for librosa
os.environ['PATH'] += os.pathsep + os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())

from transformers import (
    Wav2Vec2ForCTC, 
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
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
        
        # Paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        local_model_path = os.path.join(base_dir, '..', 'Wav2Vec2', 'checkpoint-3645')
        
        try:
            if os.path.exists(local_model_path):
                print(f"Found local Wav2Vec2 checkpoint at: {local_model_path}")
                
                tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(local_model_path, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
                feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")
                processors['wav2vec2'] = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
                models['wav2vec2'] = Wav2Vec2ForCTC.from_pretrained(local_model_path).to(device)
            else:
                print("Local Wav2Vec2 not found, downloading base model...")
                model_name = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
                processors['wav2vec2'] = Wav2Vec2Processor.from_pretrained(model_name)
                models['wav2vec2'] = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
                
            models['wav2vec2'].eval()
            print("Wav2Vec2 loaded successfully!")
            
        except Exception as e:
            print(f"Error loading Wav2Vec2: {e}")
            model_name = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
            processors['wav2vec2'] = Wav2Vec2Processor.from_pretrained(model_name)
            models['wav2vec2'] = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
            models['wav2vec2'].eval()
            
    return models['wav2vec2'], processors['wav2vec2']

def load_phowhisper_finetuned():
    """Load PhoWhisper model (Fine-tuned with LoRA)"""
    if 'phowhisper_finetuned' not in models:
        print("Loading PhoWhisper (Fine-tuned)...")
        base_model_name = "vinai/PhoWhisper-small"
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        adapter_path = os.path.join(base_dir, '..', 'PhoWhisper', 'phowhisper-finetuned-local', 'checkpoint-500')
        
        try:
            processor = WhisperProcessor.from_pretrained(base_model_name)
            model = WhisperForConditionalGeneration.from_pretrained(base_model_name)
            
            if os.path.exists(adapter_path):
                print(f"Found local PhoWhisper adapter at: {adapter_path}")
                model = PeftModel.from_pretrained(model, adapter_path)
            else:
                print("Local PhoWhisper adapter not found, using base model.")
            
            models['phowhisper_finetuned'] = model.to(device)
            processors['phowhisper_finetuned'] = processor
            models['phowhisper_finetuned'].eval()
            print("PhoWhisper (Fine-tuned) loaded successfully!")
            
        except Exception as e:
            print(f"Error loading PhoWhisper (Fine-tuned): {e}")
            # Fallback to base
            processors['phowhisper_finetuned'] = WhisperProcessor.from_pretrained(base_model_name)
            models['phowhisper_finetuned'] = WhisperForConditionalGeneration.from_pretrained(base_model_name).to(device)
            models['phowhisper_finetuned'].eval()
            
    return models['phowhisper_finetuned'], processors['phowhisper_finetuned']

def load_phowhisper_base():
    """Load PhoWhisper model (Pre-trained Base)"""
    if 'phowhisper_base' not in models:
        print("Loading PhoWhisper (Base)...")
        model_name = "vinai/PhoWhisper-small"
        processors['phowhisper_base'] = WhisperProcessor.from_pretrained(model_name)
        models['phowhisper_base'] = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        models['phowhisper_base'].eval()
        print("PhoWhisper (Base) loaded successfully!")
    return models['phowhisper_base'], processors['phowhisper_base']

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
    model, processor = load_wav2vec2()
    # Read directly with SoundFile (already 16kHz from FFmpeg)
    speech, _ = sf.read(audio_path)
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(predicted_ids)[0]

def transcribe_phowhisper(audio_path, finetuned=True):
    if finetuned:
        model, processor = load_phowhisper_finetuned()
    else:
        model, processor = load_phowhisper_base()
        
    # Read directly with SoundFile
    speech, _ = sf.read(audio_path)
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        generated_ids = model.generate(
            inputs["input_features"],
            max_length=225,
            language="vi",
            task="transcribe",
            no_repeat_ngram_size=3 if finetuned else 0,
            repetition_penalty=1.2 if finetuned else 1.0
        )
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

def transcribe_whisper(audio_path):
    model, processor = load_whisper()
    # Read directly with SoundFile
    speech, _ = sf.read(audio_path)
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        generated_ids = model.generate(
            inputs["input_features"],
            max_length=225,
            language="vi",
            task="transcribe"
        )
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

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
        
        # Convert to WAV using explicit FFmpeg subprocess for robustness
        import subprocess
        
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        print(f"Using FFmpeg at: {ffmpeg_exe}")
        
        try:
            # Command: ffmpeg -y -i input -ar 16000 -ac 1 output.wav
            cmd = [
                ffmpeg_exe, '-y',
                '-i', temp_input,
                '-ar', '16000',
                '-ac', '1',
                temp_path
            ]
            
            # Run conversion
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Converted audio to {temp_path}")
            
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg conversion failed: {e.stderr.decode()}")
            return jsonify({'error': f'Audio conversion failed: {e.stderr.decode()}'}), 400
        except Exception as e:
            print(f"General conversion error: {str(e)}")
            return jsonify({'error': f'Audio processing error: {str(e)}'}), 500
        

        # Check for silence before transcribing
        def check_silence(audio_path, threshold=0.01):
            """Check if audio is silent using RMS energy"""
            y, _ = sf.read(audio_path)
            rms = np.sqrt(np.mean(y**2))
            print(f"Audio RMS: {rms:.4f}")
            return rms < threshold

        if check_silence(temp_path):
            print("Silence detected, skipping transcription.")
            transcription = ""  # Return empty string for silence
        else:
            # Transcribe based on model choice
            if model_choice == 'wav2vec2':
                transcription = transcribe_wav2vec2(temp_path)
            elif model_choice == 'phowhisper_finetuned':
                transcription = transcribe_phowhisper(temp_path, finetuned=True)
            elif model_choice == 'phowhisper_base':
                transcription = transcribe_phowhisper(temp_path, finetuned=False)
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
