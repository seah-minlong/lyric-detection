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


def compare_models_on_test_data():
    """
    Compare baseline Whisper model with fine-tuned model on test data.
    """
    test_songs_dir = "test_data/songs"
    test_lyrics_dir = "test_data/lyrics"
    
    baseline_model = "openai/whisper-base"
    finetuned_model = "whisper-lyrics-final"
    
    # Map song files to lyric files
    test_files = {
        "fernando.mp3": "fernando.txt",
        "gimme-a-man-after-midnight.mp3": "gimme-a-man-after-midnight.txt",
        "mamma-mia.mp3": "mama-mia.txt",
        "slipping-through-my-fingers.mp3": "slipping-through-my-fingers.txt"
    }
    
    results_baseline = []
    results_finetuned = []
    
    print("\n" + "=" * 80)
    print("MODEL COMPARISON: BASELINE vs FINE-TUNED")
    print("=" * 80)
    
    for song_file, lyric_file in test_files.items():
        song_path = os.path.join(test_songs_dir, song_file)
        lyrics_path = os.path.join(test_lyrics_dir, lyric_file)
        
        if not os.path.exists(song_path):
            print(f"Warning: Song file not found: {song_path}")
            continue
        
        if not os.path.exists(lyrics_path):
            print(f"Warning: Lyrics file not found: {lyrics_path}")
            continue
        
        print(f"\n{'=' * 80}")
        print(f"Processing: {song_file}")
        print(f"{'=' * 80}")
        
        # Evaluate with baseline model
        print(f"\nEvaluating with BASELINE model ({baseline_model})...")
        baseline_result = evaluate_transcription(
            song_path=song_path,
            ground_truth_path=lyrics_path,
            whisper_model=baseline_model
        )
        if baseline_result:
            results_baseline.append(baseline_result)
            print(f"Baseline WER: {baseline_result['metrics']['word_error_rate']*100:.2f}%")
        
        # Evaluate with fine-tuned model
        print(f"\nEvaluating with FINE-TUNED model ({finetuned_model})...")
        finetuned_result = evaluate_transcription(
            song_path=song_path,
            ground_truth_path=lyrics_path,
            whisper_model=finetuned_model
        )
        if finetuned_result:
            results_finetuned.append(finetuned_result)
            print(f"Fine-tuned WER: {finetuned_result['metrics']['word_error_rate']*100:.2f}%")
        
        # Show comparison for this song
        if baseline_result and finetuned_result:
            wer_diff = baseline_result['metrics']['word_error_rate'] - finetuned_result['metrics']['word_error_rate']
            improvement = wer_diff * 100
            print(f"\n{'*' * 40}")
            print(f"WER Improvement: {improvement:+.2f}%")
            if improvement > 0:
                print("✓ Fine-tuned model performs BETTER")
            elif improvement < 0:
                print("✗ Baseline model performs BETTER")
            else:
                print("= Same performance")
            print(f"{'*' * 40}")
    
    # Print summary comparison
    print_comparison_summary(results_baseline, results_finetuned)
    
    return results_baseline, results_finetuned


def print_comparison_summary(results_baseline: List[Dict], results_finetuned: List[Dict]):
    """
    Print a summary table comparing baseline and fine-tuned models.
    """
    print("\n" + "=" * 80)
    print("SUMMARY: MODEL COMPARISON")
    print("=" * 80)
    
    if not results_baseline or not results_finetuned:
        print("Insufficient results for comparison")
        return
    
    print(f"\n{'Song':<35} {'Baseline WER':<15} {'Fine-tuned WER':<15} {'Improvement':<12}")
    print("-" * 80)
    
    total_baseline_wer = 0
    total_finetuned_wer = 0
    
    for baseline_res, finetuned_res in zip(results_baseline, results_finetuned):
        song_name = os.path.basename(baseline_res['song_path'])
        baseline_wer = baseline_res['metrics']['word_error_rate'] * 100
        finetuned_wer = finetuned_res['metrics']['word_error_rate'] * 100
        improvement = baseline_wer - finetuned_wer
        
        total_baseline_wer += baseline_wer
        total_finetuned_wer += finetuned_wer
        
        print(f"{song_name:<35} {baseline_wer:>6.2f}%{'':<8} {finetuned_wer:>6.2f}%{'':<8} {improvement:>+6.2f}%")
    
    print("-" * 80)
    
    avg_baseline_wer = total_baseline_wer / len(results_baseline)
    avg_finetuned_wer = total_finetuned_wer / len(results_finetuned)
    avg_improvement = avg_baseline_wer - avg_finetuned_wer
    
    print(f"{'AVERAGE':<35} {avg_baseline_wer:>6.2f}%{'':<8} {avg_finetuned_wer:>6.2f}%{'':<8} {avg_improvement:>+6.2f}%")
    print("=" * 80)
    
    print(f"\nOverall Performance:")
    print(f"  Baseline Model Average WER: {avg_baseline_wer:.2f}%")
    print(f"  Fine-tuned Model Average WER: {avg_finetuned_wer:.2f}%")
    print(f"  Average Improvement: {avg_improvement:+.2f}%")
    
    if avg_improvement > 0:
        relative_improvement = (avg_improvement / avg_baseline_wer) * 100
        print(f"  Relative Improvement: {relative_improvement:.2f}%")
        print(f"\n✓ Fine-tuned model shows overall improvement!")
    elif avg_improvement < 0:
        print(f"\n✗ Baseline model performs better on average")
    else:
        print(f"\n= Models show similar performance")
    
    print("=" * 80)


def main():
    """
    Main function to run model comparison on test data.
    """
    compare_models_on_test_data()


if __name__ == "__main__":
    main()
