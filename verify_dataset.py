"""
Verify dataset images exist and sample images from each category
"""
import pandas as pd
import os
from pathlib import Path
from collections import defaultdict
import random

def verify_dataset(csv_path):
    """Verify images exist and collect samples by category"""
    print(f"\n{'='*80}")
    print(f"Verifying: {csv_path}")
    print(f"{'='*80}\n")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"Total rows in CSV: {len(df)}")
    
    # Check image paths
    missing_images = []
    existing_images = []
    
    for idx, row in df.iterrows():
        img_path = row['image_path']
        if os.path.exists(img_path):
            existing_images.append((idx, row))
        else:
            missing_images.append(img_path)
    
    print(f"Existing images: {len(existing_images)}")
    print(f"Missing images: {len(missing_images)}")
    
    if missing_images:
        print(f"\n⚠️  WARNING: {len(missing_images)} images not found on disk!")
        print(f"First 10 missing images:")
        for path in missing_images[:10]:
            print(f"  - {path}")
    else:
        print("✓ All images exist on disk!")
    
    # Categorize existing images
    categories = {
        'Real': [],
        'GANs': [],
        'DMs': [],
        'deepfakes': []
    }
    
    for idx, row in existing_images:
        img_path = row['image_path']
        if '/Real/' in img_path:
            categories['Real'].append((idx, row, img_path))
        elif '/GANs/' in img_path:
            categories['GANs'].append((idx, row, img_path))
        elif '/DMs/' in img_path:
            categories['DMs'].append((idx, row, img_path))
        elif '/deepfakes/' in img_path:
            categories['deepfakes'].append((idx, row, img_path))
    
    # Display category statistics
    print(f"\n{'Category Distribution:':-^80}")
    for cat, items in categories.items():
        print(f"  {cat:15s}: {len(items):5d} images")
    
    # Sample 10 images per category for manual verification
    print(f"\n{'Sample Images for Manual Verification:':-^80}")
    samples = {}
    for cat, items in categories.items():
        if items:
            n_samples = min(10, len(items))
            sampled = random.sample(items, n_samples)
            samples[cat] = sampled
            print(f"\n{cat} ({n_samples} samples):")
            for idx, row, img_path in sampled:
                label = 'REAL' if row['target'] == 0 else 'FAKE'
                print(f"  Row {idx:4d} | {label:4s} | {row['gender']:6s} | ST:{row['skin_tone']:2d} | {img_path}")
    
    return {
        'total': len(df),
        'existing': len(existing_images),
        'missing': len(missing_images),
        'categories': {k: len(v) for k, v in categories.items()},
        'samples': samples,
        'missing_paths': missing_images
    }

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Verify both train and test datasets
    train_results = verify_dataset('dataset/train_subset_mapped.csv')
    test_results = verify_dataset('dataset/test_subset_mapped.csv')
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"\nTrain Dataset:")
    print(f"  Total: {train_results['total']}")
    print(f"  Existing: {train_results['existing']}")
    print(f"  Missing: {train_results['missing']}")
    
    print(f"\nTest Dataset:")
    print(f"  Total: {test_results['total']}")
    print(f"  Existing: {test_results['existing']}")
    print(f"  Missing: {test_results['missing']}")
    
    if train_results['missing'] > 0 or test_results['missing'] > 0:
        print(f"\n⚠️  ACTION REQUIRED: Images directory structure needs to be created or downloaded!")
        print(f"Expected structure: images/train/ and images/test/ with Real/, GANs/, DMs/, deepfakes/ subdirectories")
