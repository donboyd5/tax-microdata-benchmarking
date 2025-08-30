#!/usr/bin/env python3
"""
Simple voice input script for terminal use with Claude Code.
Usage: python3 voice_input.py [duration_seconds]
"""
import sys
import subprocess
import tempfile
import os
import whisper

def record_audio(duration=10):
    """Record audio for specified duration."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        temp_filename = temp_audio.name
    
    print(f"🎤 Recording for {duration} seconds... Speak now!")
    
    # Record audio using arecord
    try:
        subprocess.run([
            'arecord', 
            '-f', 'cd',  # CD quality
            '-t', 'wav',
            '-d', str(duration),
            temp_filename
        ], check=True, capture_output=True)
        print("✅ Recording complete!")
        return temp_filename
    except subprocess.CalledProcessError as e:
        print(f"❌ Recording failed: {e}")
        return None

def transcribe_audio(audio_file):
    """Transcribe audio using Whisper."""
    print("🤖 Transcribing audio...")
    
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_file)
        return result["text"].strip()
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        return None

def main():
    # Get recording duration from command line argument, default to 10 seconds
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    
    # Record audio
    audio_file = record_audio(duration)
    if not audio_file:
        return 1
    
    try:
        # Transcribe audio
        text = transcribe_audio(audio_file)
        if text:
            print(f"\n📝 Transcribed text:")
            print(text)
            print()
            
            # Copy to clipboard if available
            try:
                subprocess.run(['xclip', '-selection', 'clipboard'], 
                             input=text.encode(), check=True)
                print("📋 Text copied to clipboard!")
            except (subprocess.CalledProcessError, FileNotFoundError):
                try:
                    subprocess.run(['wl-copy'], input=text.encode(), check=True)
                    print("📋 Text copied to clipboard!")
                except (subprocess.CalledProcessError, FileNotFoundError):
                    print("ℹ️  Install xclip or wl-clipboard to auto-copy text")
        else:
            return 1
            
    finally:
        # Clean up temporary file
        if os.path.exists(audio_file):
            os.unlink(audio_file)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())