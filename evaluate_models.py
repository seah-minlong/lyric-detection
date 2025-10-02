"""
Gemini was used for this evaluation code

Comprehensive model evaluation script for Chapter 5 - Model Performance and Evaluation

This script:
1. Evaluates both baseline (openai/whisper-small) and fine-tuned models
2. Runs evaluation on the full test set
3. Computes aggregate WER metrics
4. Outputs formatted results for report tables
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
import evaluate
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import torchaudio
from tqdm import tqdm
import re


def normalize_text(text: str) -> str:
    """Clean up lyrics text - same normalization as training"""
    # Remove structural markers like [Intro], [Verse 1], [Chorus], etc.
    text = re.sub(r'\[.*?\]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_lyrics_file(lyrics_path: str) -> str:
    """Load and normalize lyrics from file"""
    try:
        with open(lyrics_path, 'r', encoding='utf-8') as f:
            lyrics = f.read()
        return normalize_text(lyrics)
    except Exception as e:
        print(f"Error loading {lyrics_path}: {e}")
        return None


def transcribe_audio(audio_path: str, model, processor, device) -> str:
    """Transcribe audio file using given model and processor"""
    try:
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample to 16kHz
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        audio_array = waveform.squeeze().numpy()
        
        # Extract features
        input_features = processor.feature_extractor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(device)
        
        # Generate transcription
        with torch.no_grad():
            predicted_ids = model.generate(input_features, max_length=448)
        
        # Decode
        transcription = processor.tokenizer.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]
        
        return normalize_text(transcription)
        
    except Exception as e:
        print(f"Error transcribing {audio_path}: {e}")
        return None


def evaluate_on_test_set(
    test_songs: List[str],
    test_lyrics: List[str],
    model_name: str,
    model_path: str = None
) -> Dict:
    """
    Evaluate a model on the test set
    
    Args:
        test_songs: List of paths to test audio files (vocal tracks)
        test_lyrics: List of paths to corresponding ground truth lyrics
        model_name: Name/identifier for the model
        model_path: Path to fine-tuned model (None for baseline)
    
    Returns:
        Dictionary with evaluation results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Device: {device}")
    print(f"Test samples: {len(test_songs)}")
    print(f"{'='*60}\n")
    
    # Load model and processor
    if model_path and os.path.exists(model_path):
        print(f"Loading fine-tuned model from: {model_path}")
        processor = WhisperProcessor.from_pretrained(model_path)
        model = WhisperForConditionalGeneration.from_pretrained(model_path)
    else:
        print(f"Loading baseline model: openai/whisper-small")
        processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        # Set same config as training
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    
    model = model.to(device)
    model.eval()
    
    # Load WER metric
    metric_wer = evaluate.load("wer")
    
    # Collect predictions and references
    predictions = []
    references = []
    individual_results = []
    
    for song_path, lyrics_path in tqdm(zip(test_songs, test_lyrics), total=len(test_songs), desc="Transcribing"):
        # Load ground truth
        ground_truth = load_lyrics_file(lyrics_path)
        if ground_truth is None:
            continue
        
        # Transcribe
        prediction = transcribe_audio(song_path, model, processor, device)
        if prediction is None:
            continue
        
        predictions.append(prediction)
        references.append(ground_truth)
        
        # Calculate per-sample WER
        sample_wer = 100 * metric_wer.compute(predictions=[prediction], references=[ground_truth])
        
        individual_results.append({
            'song': os.path.basename(song_path),
            'wer': sample_wer,
            'prediction': prediction,
            'reference': ground_truth
        })
    
    # Calculate aggregate WER
    if len(predictions) > 0:
        aggregate_wer = 100 * metric_wer.compute(predictions=predictions, references=references)
    else:
        aggregate_wer = 100.0
    
    results = {
        'model_name': model_name,
        'aggregate_wer': aggregate_wer,
        'num_samples': len(predictions),
        'individual_results': individual_results
    }
    
    return results


def find_test_files(data_dir: str = "data") -> Tuple[List[str], List[str]]:
    """
    Find test audio and lyrics files
    Assumes vocal tracks are in output_demucs/ or processed_vocals/
    """
    songs_dir = Path(data_dir) / "songs"
    lyrics_dir = Path(data_dir) / "lyrics"
    
    # Look for vocal tracks (from Demucs output)
    vocal_dirs = [
        Path("output_demucs") / "htdemucs",
        Path("processed_vocals") / "htdemucs"
    ]
    
    test_songs = []
    test_lyrics = []
    
    # Find all vocal tracks
    for vocal_dir in vocal_dirs:
        if vocal_dir.exists():
            for song_folder in vocal_dir.iterdir():
                if song_folder.is_dir():
                    vocal_path = song_folder / "vocals.wav"
                    if vocal_path.exists():
                        # Find corresponding lyrics
                        song_name = song_folder.name
                        lyrics_path = lyrics_dir / f"{song_name}.txt"
                        
                        if lyrics_path.exists():
                            test_songs.append(str(vocal_path))
                            test_lyrics.append(str(lyrics_path))
    
    return test_songs, test_lyrics


def print_report_tables(baseline_results: Dict, finetuned_results: Dict):
    """Print formatted tables for Chapter 5"""
    
    print("\n" + "="*70)
    print("CHAPTER 5 - MODEL PERFORMANCE AND EVALUATION")
    print("="*70)
    
    print("\n5.1 Training Results")
    print("-" * 70)
    print("\nFinal evaluation on the test set:\n")
    print("| Metric                  | Score       |")
    print("|-------------------------|-------------|")
    print(f"| Word Error Rate (WER)   | {finetuned_results['aggregate_wer']:.2f}%      |")
    print()
    
    print("\n5.2 Comparison with Baseline Model")
    print("-" * 70)
    print("\nPerformance comparison:\n")
    print("| Model                   | Word Error Rate (WER) |")
    print("|-------------------------|-----------------------|")
    print(f"| Baseline whisper-small  | {baseline_results['aggregate_wer']:.2f}%                 |")
    print(f"| Fine-tuned whisper-small| {finetuned_results['aggregate_wer']:.2f}%                 |")
    print()
    
    # Calculate improvement
    improvement = baseline_results['aggregate_wer'] - finetuned_results['aggregate_wer']
    improvement_pct = (improvement / baseline_results['aggregate_wer']) * 100
    
    print(f"\n📊 Analysis:")
    print(f"   - Absolute WER reduction: {improvement:.2f} percentage points")
    print(f"   - Relative improvement: {improvement_pct:.1f}%")
    print(f"   - Test set size: {finetuned_results['num_samples']} samples")
    
    print("\n" + "="*70)
    
    # Per-song breakdown
    print("\n\n📋 Per-Song Results:")
    print("-" * 70)
    print(f"\n{'Song':<30} {'Baseline WER':<15} {'Fine-tuned WER':<15} {'Δ':<10}")
    print("-" * 70)
    
    for baseline_item, finetuned_item in zip(
        baseline_results['individual_results'],
        finetuned_results['individual_results']
    ):
        song_name = baseline_item['song']
        baseline_wer = baseline_item['wer']
        finetuned_wer = finetuned_item['wer']
        delta = baseline_wer - finetuned_wer
        
        print(f"{song_name:<30} {baseline_wer:>6.2f}%         {finetuned_wer:>6.2f}%         {delta:>+6.2f}%")
    
    print("-" * 70)


def save_detailed_results(baseline_results: Dict, finetuned_results: Dict, output_file: str = "evaluation_results.json"):
    """Save detailed results to JSON file"""
    results = {
        'baseline': baseline_results,
        'finetuned': finetuned_results,
        'summary': {
            'baseline_wer': baseline_results['aggregate_wer'],
            'finetuned_wer': finetuned_results['aggregate_wer'],
            'improvement': baseline_results['aggregate_wer'] - finetuned_results['aggregate_wer'],
            'num_samples': finetuned_results['num_samples']
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Detailed results saved to: {output_file}")


def main():
    """Main evaluation script"""
    
    # Find test files
    print("🔍 Finding test files...")
    test_songs, test_lyrics = find_test_files()
    
    if len(test_songs) == 0:
        print("❌ No test files found!")
        print("Make sure you have:")
        print("  1. Vocal tracks in output_demucs/htdemucs/ or processed_vocals/htdemucs/")
        print("  2. Corresponding lyrics in data/lyrics/")
        return
    
    print(f"✅ Found {len(test_songs)} test samples")
    
    # Evaluate baseline model
    baseline_results = evaluate_on_test_set(
        test_songs=test_songs,
        test_lyrics=test_lyrics,
        model_name="Baseline (openai/whisper-small)",
        model_path=None
    )
    
    # Evaluate fine-tuned model
    finetuned_model_path = "./whisper-lyrics-final"
    if not os.path.exists(finetuned_model_path):
        print(f"\n⚠️  Fine-tuned model not found at: {finetuned_model_path}")
        print("Using './whisper-lyrics-model' instead...")
        finetuned_model_path = "./whisper-lyrics-model"
    
    finetuned_results = evaluate_on_test_set(
        test_songs=test_songs,
        test_lyrics=test_lyrics,
        model_name="Fine-tuned whisper-small",
        model_path=finetuned_model_path
    )
    
    # Print report tables
    print_report_tables(baseline_results, finetuned_results)
    
    # Save detailed results
    save_detailed_results(baseline_results, finetuned_results)
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
