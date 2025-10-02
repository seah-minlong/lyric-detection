import os
import difflib
from model import transcribe_song
import re
from typing import Dict, List, Tuple


def load_ground_truth_lyrics(lyrics_file_path: str) -> str:
    try:
        with open(lyrics_file_path, 'r', encoding='utf-8') as f:
            lyrics = f.read().strip()
        return lyrics
    except FileNotFoundError:
        print(f"Error: Ground truth file not found: {lyrics_file_path}")
        return None
    except Exception as e:
        print(f"Error reading ground truth file: {e}")
        return None


def normalize_text(text: str) -> str:
    if not text:
        return ""
    
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def calculate_word_accuracy(predicted: str, ground_truth: str) -> Dict[str, float]:
    pred_normalized = normalize_text(predicted)
    truth_normalized = normalize_text(ground_truth)
    
    pred_words = pred_normalized.split()
    truth_words = truth_normalized.split()
    
    sequence_matcher = difflib.SequenceMatcher(None, pred_words, truth_words)
    similarity_ratio = sequence_matcher.ratio()
    
    opcodes = sequence_matcher.get_opcodes()
    substitutions = insertions = deletions = 0
    
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'replace':
            substitutions += max(i2 - i1, j2 - j1)
        elif tag == 'delete':
            deletions += i2 - i1
        elif tag == 'insert':
            insertions += j2 - j1
    
    total_errors = substitutions + insertions + deletions
    wer = total_errors / len(truth_words) if len(truth_words) > 0 else 1.0
    word_accuracy = max(0, 1 - wer)
    
    return {
        'similarity_ratio': similarity_ratio,
        'word_accuracy': word_accuracy,
        'word_error_rate': wer,
        'total_predicted_words': len(pred_words),
        'total_ground_truth_words': len(truth_words),
        'substitutions': substitutions,
        'insertions': insertions,
        'deletions': deletions
    }


def get_detailed_diff(predicted: str, ground_truth: str) -> List[str]:
    pred_normalized = normalize_text(predicted)
    truth_normalized = normalize_text(ground_truth)
    
    pred_words = pred_normalized.split()
    truth_words = truth_normalized.split()
    
    diff = difflib.unified_diff(
        truth_words, pred_words,
        fromfile='Ground Truth',
        tofile='Transcribed',
        lineterm=''
    )
    
    return list(diff)


def evaluate_transcription(song_path: str, ground_truth_path: str, 
                         whisper_model: str = "openai/whisper-base",
                         output_dir: str = "output_demucs") -> Dict:
    print("Starting transcription evaluation...")
    print(f"Song: {song_path}")
    print(f"Ground Truth: {ground_truth_path}")
    print(f"Model: {whisper_model}")
    
    ground_truth = load_ground_truth_lyrics(ground_truth_path)
    if ground_truth is None:
        return None
    
    print("\nRunning transcription...")
    predicted_lyrics = transcribe_song(song_path, whisper_model)
    
    if predicted_lyrics is None:
        print("Transcription failed!")
        return None
    
    print("Calculating metrics...")
    metrics = calculate_word_accuracy(predicted_lyrics, ground_truth)
    
    diff = get_detailed_diff(predicted_lyrics, ground_truth)
    
    results = {
        'song_path': song_path,
        'ground_truth_path': ground_truth_path,
        'whisper_model': whisper_model,
        'predicted_lyrics': predicted_lyrics,
        'ground_truth_lyrics': ground_truth,
        'metrics': metrics,
        'detailed_diff': diff
    }
    
    return results


def print_evaluation_results(results: Dict):
    if results is None:
        print("No results to display")
        return
    
    metrics = results['metrics']
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"Song: {os.path.basename(results['song_path'])}")
    print(f"Model: {results['whisper_model']}")
    print(f"\nWord Error Rate (WER): {metrics['word_error_rate']*100:.2f}%")
    print(f"Reference words: {metrics['total_ground_truth_words']}")
    print(f"Errors: {metrics['substitutions'] + metrics['insertions'] + metrics['deletions']} (sub: {metrics['substitutions']}, ins: {metrics['insertions']}, del: {metrics['deletions']})")
    
    print("=" * 60)


def main():
    song_path = "data/songs/slimshady.mp3" 
    ground_truth_path = "data/lyrics/slimshady.txt"
    # model = "whisper-lyrics-final"

    if not os.path.exists(song_path):
        print(f"Error: Song file not found: {song_path}")
        return
    
    if not os.path.exists(ground_truth_path):
        print(f"Error: Ground truth file not found: {ground_truth_path}")
        return
    
    results = evaluate_transcription(
        song_path=song_path,
        ground_truth_path=ground_truth_path,
        # whisper_model=model
    )
    
    print_evaluation_results(results)


if __name__ == "__main__":
    main()
