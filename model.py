import os
import subprocess
import torch
from transformers import pipeline


def separate_vocals(input_song_path, output_dir="output_demucs"):
    """
    Separate vocals from a song using Demucs.
    
    Args:
        input_song_path (str): Path to the input song file
        output_dir (str): Directory to save the separated vocals
        
    Returns:
        str: Path to the separated vocal track
        
    Raises:
        subprocess.CalledProcessError: If Demucs separation fails
        FileNotFoundError: If the vocal track is not found after separation
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Separating vocals from: {input_song_path}")
    try:
        subprocess.run([
            "python", "-m", "demucs", "--two-stems", "vocals",
            "-o", output_dir, input_song_path
        ], check=True)
        print("Demucs separation complete.")
    except subprocess.CalledProcessError as e:
        print(f"Error during Demucs separation: {e}")
        raise
    
    # Find the path to the generated vocal track
    song_name_without_ext = os.path.splitext(os.path.basename(input_song_path))[0]
    vocal_track_path = os.path.join(output_dir, "htdemucs", song_name_without_ext, "vocals.wav")
    
    if not os.path.exists(vocal_track_path):
        error_msg = f"Could not find the vocal track at the expected path: {vocal_track_path}"
        print(f"❌ Error: {error_msg}")
        print(f"Please check the '{output_dir}' folder to see what was generated.")
        raise FileNotFoundError(error_msg)
    
    print(f"🎤 Vocal track found at: {vocal_track_path}")
    return vocal_track_path


def transcribe_audio(audio_path, model_name="openai/whisper-base", return_timestamps=True):
    """
    Transcribe audio using Whisper model.
    
    Args:
        audio_path (str): Path to the audio file to transcribe
        model_name (str): Whisper model to use (default: "openai/whisper-base")
        return_timestamps (bool): Whether to return timestamps in the transcription
        
    Returns:
        dict: Transcription result containing text and optionally timestamps
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device for Whisper: {device}")
    
    print(f"Loading the Whisper model: {model_name}...")
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=device
    )
    print("Model loaded successfully!")
    
    print("\nTranscribing the audio... (This may take a while depending on audio length)")
    transcription_result = asr_pipeline(audio_path, return_timestamps=return_timestamps, generate_kwargs={"language": "en"})
    
    return transcription_result


def transcribe_song(input_song_path, whisper_model="openai/whisper-base", output_dir="output_demucs"):
    """
    Complete pipeline: separate vocals from song and transcribe them.
    
    Args:
        input_song_path (str): Path to the input song file
        whisper_model (str): Whisper model to use for transcription
        output_dir (str): Directory to save the separated vocals
        
    Returns:
        str: Transcribed lyrics
    """
    try:
        # Step 1: Separate vocals
        vocal_track_path = separate_vocals(input_song_path, output_dir)
        
        # Step 2: Transcribe vocals
        transcription_result = transcribe_audio(vocal_track_path, whisper_model)
        
        lyrics = transcription_result["text"]
        
        print("\n--- 📜 DETECTED LYRICS ---")
        print(lyrics)
        
        return lyrics
        
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error in transcription pipeline: {e}")
        return None


def main():
    song_path = "songs/eminem_the-real-slim-shady.mp3"  
    
    # Run the transcription pipeline
    lyrics = transcribe_song(song_path)
    
    if lyrics:
        print("\n✅ Transcription completed successfully!")
    else:
        print("\n❌ Transcription failed!")


if __name__ == "__main__":
    main()