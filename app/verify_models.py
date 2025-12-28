import sys
import os

# Add app directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import (
    load_wav2vec2, 
    load_phowhisper_finetuned, 
    load_phowhisper_base, 
    load_whisper
)

print("=== Starting Model Verification ===")

print("\n--- Test 1: Wav2Vec2 (Local) ---")
try:
    load_wav2vec2()
    print("✅ Wav2Vec2 Loaded Successfully")
except Exception as e:
    print(f"❌ Wav2Vec2 Failed: {e}")

print("\n--- Test 2: PhoWhisper Finetuned (Local) ---")
try:
    load_phowhisper_finetuned()
    print("✅ PhoWhisper Finetuned Loaded Successfully")
except Exception as e:
    print(f"❌ PhoWhisper Finetuned Failed: {e}")

print("\n--- Test 3: PhoWhisper Base (Pretrained) ---")
try:
    load_phowhisper_base()
    print("✅ PhoWhisper Base Loaded Successfully")
except Exception as e:
    print(f"❌ PhoWhisper Base Failed: {e}")

print("\n--- Test 4: OpenAI Whisper (Standard) ---")
try:
    load_whisper()
    print("✅ OpenAI Whisper Loaded Successfully")
except Exception as e:
    print(f"❌ OpenAI Whisper Failed: {e}")

print("\n=== Verification Complete ===")
