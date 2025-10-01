import os
import subprocess
import torch
from transformers import pipeline

DEMUCS_OUTPUT_FOLDER = "htdemucs"

def extract_vocals_from_song(song_file, output_folder="output_demucs"):
    """Separates vocals from background music using Demucs"""
    
    if not os.path.exists(song_file):
        raise FileNotFoundError(f"Song file not found: {song_file}")
    
    # create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Processing: {os.path.basename(song_file)}")
    
    # run demucs vocal separation
    cmd = ["python", "-m", "demucs", "--two-stems", "vocals", "-o", output_folder, song_file]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Demucs failed with error:", result.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd)
    
    # figure out where demucs saved the vocal track
    song_basename = os.path.splitext(os.path.basename(song_file))[0]
    vocals_path = os.path.join(output_folder, DEMUCS_OUTPUT_FOLDER, song_basename, "vocals.wav")
    
    # sanity check that the file actually exists
    if not os.path.exists(vocals_path):
        print(f"Warning: Expected vocals at {vocals_path} but file doesn't exist")
        # check what's actually in the output directory
        base_dir = os.path.join(output_folder, DEMUCS_OUTPUT_FOLDER, song_basename)
        if os.path.exists(base_dir):
            files = os.listdir(base_dir)
            print(f"Found files: {files}")
        raise FileNotFoundError(f"Vocals file missing: {vocals_path}")
    
    print(f"Vocals extracted to: {vocals_path}")
    return vocals_path


def get_lyrics_from_audio(audio_file, model="openai/whisper-base"):
    """Convert audio to text using Whisper"""
    
    # check if we can use GPU - whisper is much faster with CUDA
    use_gpu = torch.cuda.is_available()
    device = "cuda:0" if use_gpu else "cpu"
    
    if use_gpu:
        print("Using GPU acceleration")
    else:
        print("Using CPU (this might be slow)")
    
    # load whisper model
    print(f"Loading {model}...")
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=model,
        device=device
    )
    
    # transcribe the vocals
    print("Converting speech to text...")
    result = transcriber(
        audio_file, 
        return_timestamps=True,
        generate_kwargs={"language": "en"}
    )
    
    return result


def transcribe_song(song_path, model="openai/whisper-base"):
    """Extract lyrics from a song file"""
    
    try:
        print(f"\n=== Processing: {os.path.basename(song_path)} ===")
        
        # first, separate the vocals from the music
        vocals_file = extract_vocals_from_song(song_path)
        
        # then convert the vocals to text  
        transcription = get_lyrics_from_audio(vocals_file, model)
        detected_lyrics = transcription["text"]
        
        print("\n=== LYRICS ===")
        print(detected_lyrics)
        print("=" * 50)
        
        return detected_lyrics
        
    except Exception as e:
        print(f"Something went wrong: {e}")
        return None


def main():
    # test with a sample song
    test_song = "songs/eminem_the-real-slim-shady.mp3"
    
    lyrics = transcribe_song(test_song)
    
    if lyrics:
        print("\nDone!")
    else:
        print("\nFailed to extract lyrics")


if __name__ == "__main__":
    main()