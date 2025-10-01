import os
import difflib
from model import transcribe_song
import re
from typing import Dict, List, Tuple


def load_ground_truth_lyrics(lyrics_file_path: str) -> str:
    """
    Load ground truth lyrics from a text file.
    
    Args:
        lyrics_file_path (str): Path to the text file containing correct lyrics
        
    Returns:
        str: Ground truth lyrics
    """
    try:
        with open(lyrics_file_path, 'r', encoding='utf-8') as f:
            lyrics = f.read().strip()
        return lyrics
    except FileNotFoundError:
        print(f"❌ Error: Ground truth file not found: {lyrics_file_path}")
        return None
    except Exception as e:
        print(f"❌ Error reading ground truth file: {e}")
        return None


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison by removing extra whitespace, 
    converting to lowercase, and removing punctuation.
    
    Args:
        text (str): Input text to normalize
        
    Returns:
        str: Normalized text
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and special characters, keep only alphanumeric and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Replace multiple whitespaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def calculate_word_accuracy(predicted: str, ground_truth: str) -> Dict[str, float]:
    """
    Calculate word-level accuracy metrics.
    
    Args:
        predicted (str): Transcribed lyrics
        ground_truth (str): Correct lyrics
        
    Returns:
        dict: Dictionary containing accuracy metrics
    """
    # Normalize both texts
    pred_normalized = normalize_text(predicted)
    truth_normalized = normalize_text(ground_truth)
    
    # Split into words
    pred_words = pred_normalized.split()
    truth_words = truth_normalized.split()
    
    # Calculate sequence matcher ratio
    sequence_matcher = difflib.SequenceMatcher(None, pred_words, truth_words)
    similarity_ratio = sequence_matcher.ratio()
    
    # Calculate word error rate (WER)
    # WER = (substitutions + insertions + deletions) / total_words_in_reference
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
    """
    Get a detailed word-by-word comparison.
    
    Args:
        predicted (str): Transcribed lyrics
        ground_truth (str): Correct lyrics
        
    Returns:
        list: List of difference strings for display
    """
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
    """
    Evaluate transcription accuracy against ground truth lyrics.
    
    Args:
        song_path (str): Path to the song file
        ground_truth_path (str): Path to the ground truth lyrics file
        whisper_model (str): Whisper model to use
        output_dir (str): Directory for demucs output
        
    Returns:
        dict: Evaluation results
    """
    print("🎵 Starting Transcription Evaluation")
    print("=" * 50)
    print(f"Song: {song_path}")
    print(f"Ground Truth: {ground_truth_path}")
    print(f"Model: {whisper_model}")
    print("=" * 50)
    
    # Load ground truth
    ground_truth = load_ground_truth_lyrics(ground_truth_path)
    if ground_truth is None:
        return None
    
    # Transcribe the song
    print("\n🔄 Running transcription...")
    predicted_lyrics = transcribe_song(song_path, whisper_model, output_dir)
    
    if predicted_lyrics is None:
        print("❌ Transcription failed!")
        return None
    
    # Calculate accuracy metrics
    print("\n📊 Calculating accuracy metrics...")
    metrics = calculate_word_accuracy(predicted_lyrics, ground_truth)
    
    # Get detailed comparison
    diff = get_detailed_diff(predicted_lyrics, ground_truth)
    
    # Prepare results
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
    """
    Print formatted evaluation results.
    
    Args:
        results (dict): Evaluation results from evaluate_transcription
    """
    if results is None:
        print("❌ No results to display")
        return
    
    metrics = results['metrics']
    
    print("\n" + "=" * 60)
    print("📈 EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"📁 Song: {os.path.basename(results['song_path'])}")
    print(f"🤖 Model: {results['whisper_model']}")
    print()
    
    print("📊 ACCURACY METRICS:")
    print(f"  • Similarity Ratio: {metrics['similarity_ratio']:.3f} ({metrics['similarity_ratio']*100:.1f}%)")
    print(f"  • Word Accuracy: {metrics['word_accuracy']:.3f} ({metrics['word_accuracy']*100:.1f}%)")
    print(f"  • Word Error Rate: {metrics['word_error_rate']:.3f} ({metrics['word_error_rate']*100:.1f}%)")
    print()
    
    print("📝 WORD COUNT:")
    print(f"  • Ground Truth Words: {metrics['total_ground_truth_words']}")
    print(f"  • Transcribed Words: {metrics['total_predicted_words']}")
    print()
    
    print("🔍 ERROR BREAKDOWN:")
    print(f"  • Substitutions: {metrics['substitutions']}")
    print(f"  • Insertions: {metrics['insertions']}")
    print(f"  • Deletions: {metrics['deletions']}")
    print()
    
    print("📜 TRANSCRIBED LYRICS:")
    print("-" * 40)
    print(results['predicted_lyrics'])
    print("-" * 40)
    
    # Show detailed diff if there are differences
    if metrics['similarity_ratio'] < 1.0:
        print("\n🔍 DETAILED DIFFERENCES:")
        print("(+ = added in transcription, - = missing from transcription)")
        print("-" * 50)
        for line in results['detailed_diff'][:20]:  # Limit to first 20 lines
            print(line)
        if len(results['detailed_diff']) > 20:
            print(f"... ({len(results['detailed_diff']) - 20} more lines)")
    
    print("=" * 60)


def main():
    """
    Main function to run evaluation with example files.
    """
    song_path = "data/songs/slimshady.mp3"  # Update this path
    ground_truth_path = "data/lyrics/slimshady.txt"  # Create this file with correct lyrics
    model = "whisper-lyrics-final"

    # Check if files exist
    if not os.path.exists(song_path):
        print(f"❌ Song file not found: {song_path}")
        print("Please update the song_path in the main() function")
        return
    
    if not os.path.exists(ground_truth_path):
        print(f"❌ Ground truth file not found: {ground_truth_path}")
        print("Please create a text file with the correct lyrics")
        print(f"Expected location: {ground_truth_path}")
        return
    
    # Run evaluation
    results = evaluate_transcription(
        song_path=song_path,
        ground_truth_path=ground_truth_path,
        whisper_model=model
    )
    
    # Print results
    print_evaluation_results(results)


if __name__ == "__main__":
    main()
