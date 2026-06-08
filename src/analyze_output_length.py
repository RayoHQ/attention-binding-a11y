"""Analyze correlation between EB* and generation output length."""

import json
from pathlib import Path
import numpy as np
from scipy import stats
import pandas as pd

def load_behavioral_data():
    """Load behavioral data with generation outputs."""
    behavioral_dir = Path('data/results/behavioral_expanded_100')
    
    all_data = []
    for f in sorted(behavioral_dir.glob('*.jsonl')):
        with open(f) as fp:
            for line in fp:
                record = json.loads(line)
                # Only keep generation tasks
                if record.get('task') == 'generation':
                    all_data.append(record)
    
    return all_data

def load_binding_data():
    """Load binding metrics."""
    binding_dir = Path('data/results/binding_expanded_100')
    
    binding_dict = {}
    for f in sorted(binding_dir.glob('*.jsonl')):
        with open(f) as fp:
            for line in fp:
                record = json.loads(line)
                # Create key for matching
                key = (record['model'], record['checkpoint'], record['term'], record['prompt_id'])
                binding_dict[key] = record
    
    return binding_dict

def compute_output_length(behavioral_record):
    """Extract output length from behavioral record."""
    # Check different possible fields for generated text
    if 'generated_text' in behavioral_record:
        text = behavioral_record['generated_text']
    elif 'completion' in behavioral_record:
        text = behavioral_record['completion']
    elif 'output' in behavioral_record:
        text = behavioral_record['output']
    else:
        return None
    
    # Count tokens (approximate with whitespace split)
    tokens = text.strip().split()
    return len(tokens)

def main():
    print("=" * 80)
    print("EB* vs OUTPUT LENGTH CORRELATION ANALYSIS")
    print("=" * 80)
    
    # Load data
    print("\nLoading behavioral data...")
    behavioral_data = load_behavioral_data()
    print(f"Loaded {len(behavioral_data)} generation records")
    
    # Check first record for structure
    if behavioral_data:
        print("\nFirst record keys:", list(behavioral_data[0].keys()))
        
        # Try to compute length
        sample_length = compute_output_length(behavioral_data[0])
        if sample_length is None:
            print("\n⚠️ No generation output found in behavioral records")
            print("Available keys:", list(behavioral_data[0].keys()))
            
            # Alternative: use merged CSV with prompt templates
            print("\nTrying merged CSV approach...")
            df = pd.read_csv('data/results/merged_100.csv')
            
            # Use prompt template length as proxy
            df['prompt_length'] = df['prompt_template_bind'].str.len()
            
            # Filter to generation tasks only
            gen_df = df[df['task'] == 'generation'].copy()
            
            print(f"\nGeneration records: {len(gen_df)}")
            print(f"EB* range: {gen_df['eb_star'].min():.3f} - {gen_df['eb_star'].max():.3f}")
            print(f"Prompt length range: {gen_df['prompt_length'].min()} - {gen_df['prompt_length'].max()}")
            
            # Compute correlation
            correlation = stats.spearmanr(gen_df['eb_star'], gen_df['prompt_length'])
            
            print("\n" + "=" * 80)
            print("CORRELATION RESULTS (EB* vs Prompt Length)")
            print("=" * 80)
            print(f"Spearman ρ: {correlation.statistic:.3f}")
            print(f"P-value: {correlation.pvalue:.4f}")
            print(f"N observations: {len(gen_df)}")
            
            if abs(correlation.statistic) < 0.1:
                print("\n✅ Negligible correlation (|ρ| < 0.1)")
                print("Output length is not a confounding factor for EB*")
            elif correlation.pvalue > 0.05:
                print("\n✅ Not statistically significant (p > 0.05)")
                print("No evidence of length confound")
            else:
                print(f"\n⚠️ Significant correlation detected")
            
            # Per-term analysis
            print("\n" + "=" * 80)
            print("PER-TERM CORRELATIONS")
            print("=" * 80)
            
            for term in sorted(gen_df['term'].unique()):
                term_df = gen_df[gen_df['term'] == term]
                if len(term_df) > 10:
                    corr = stats.spearmanr(term_df['eb_star'], term_df['prompt_length'])
                    print(f"{term:20s}: ρ = {corr.statistic:+.3f}, p = {corr.pvalue:.4f}, n = {len(term_df)}")
            
            return
    
    print("\nLoading binding data...")
    binding_dict = load_binding_data()
    print(f"Loaded {len(binding_dict)} binding records")
    
    # Match and analyze
    matched_data = []
    for beh in behavioral_data:
        key = (beh['model'], beh['checkpoint'], beh['term'], beh['prompt_id'])
        
        if key in binding_dict:
            length = compute_output_length(beh)
            if length is not None:
                matched_data.append({
                    'eb_star': binding_dict[key]['eb_star'],
                    'output_length': length,
                    'term': beh['term'],
                    'model': beh['model'],
                    'checkpoint': beh['checkpoint']
                })
    
    print(f"\nMatched {len(matched_data)} records with both EB* and output length")
    
    if not matched_data:
        print("\n❌ Could not match behavioral and binding data")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(matched_data)
    
    # Overall correlation
    correlation = stats.spearmanr(df['eb_star'], df['output_length'])
    
    print("\n" + "=" * 80)
    print("CORRELATION RESULTS")
    print("=" * 80)
    print(f"Spearman ρ: {correlation.statistic:.3f}")
    print(f"P-value: {correlation.pvalue:.4f}")
    print(f"N observations: {len(df)}")
    
    print(f"\nEB* statistics:")
    print(f"  Mean: {df['eb_star'].mean():.3f}")
    print(f"  Std:  {df['eb_star'].std():.3f}")
    print(f"  Range: {df['eb_star'].min():.3f} - {df['eb_star'].max():.3f}")
    
    print(f"\nOutput length statistics:")
    print(f"  Mean: {df['output_length'].mean():.1f} tokens")
    print(f"  Std:  {df['output_length'].std():.1f} tokens")
    print(f"  Range: {df['output_length'].min()} - {df['output_length'].max()} tokens")
    
    if abs(correlation.statistic) < 0.1:
        print("\n✅ Negligible correlation (|ρ| < 0.1)")
        print("Output length is not a confounding factor for EB*")
    elif correlation.pvalue > 0.05:
        print("\n✅ Not statistically significant (p > 0.05)")
        print("No evidence of length confound")
    else:
        print(f"\n⚠️ Significant correlation detected")

if __name__ == "__main__":
    main()
