"""
Comprehensive Dataset Verification Tool
- Checks if images exist on disk at CSV-specified paths
- Samples ~10 images per category for visual inspection
- Generates detailed report
"""
import pandas as pd
import os
from pathlib import Path
from collections import defaultdict
import random


def get_category_from_path(img_path):
    """Extract category (Real/GANs/DMs/deepfakes) from image path"""
    parts = img_path.split('/')
    for i, part in enumerate(parts):
        if part in ['Real', 'GANs', 'DMs', 'deepfakes']:
            return part
    return 'Unknown'


def get_subcategory_from_path(img_path):
    """Extract subcategory (e.g., FFHQ, StyleGAN2, dfdc) from image path"""
    parts = img_path.split('/')
    for i, part in enumerate(parts):
        if part in ['Real', 'GANs', 'DMs', 'deepfakes']:
            if i + 1 < len(parts):
                return parts[i + 1]
    return 'Unknown'


def verify_and_sample_dataset(csv_path, dataset_name="Dataset"):
    """
    Main verification function
    """
    print(f"\n{'='*100}")
    print(f" VERIFICATION REPORT: {dataset_name}")
    print(f" CSV: {csv_path}")
    print(f"{'='*100}\n")
    
    # Check if CSV exists
    if not os.path.exists(csv_path):
        print(f"❌ ERROR: CSV file not found: {csv_path}")
        return None
    
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"📊 Total entries in CSV: {len(df):,}")
    
    # Analyze CSV structure
    print(f"\n📋 CSV Schema:")
    for col in df.columns:
        print(f"   - {col}: {df[col].dtype}")
    
    # Check target distribution
    if 'target' in df.columns:
        target_dist = df['target'].value_counts().sort_index()
        print(f"\n🎯 Target Distribution:")
        print(f"   - Real (0): {target_dist.get(0, 0):,} ({target_dist.get(0, 0)/len(df)*100:.1f}%)")
        print(f"   - Fake (1): {target_dist.get(1, 0):,} ({target_dist.get(1, 0)/len(df)*100:.1f}%)")
    
    # Check demographic distribution
    if 'gender' in df.columns:
        gender_dist = df['gender'].value_counts()
        print(f"\n👤 Gender Distribution:")
        for gender, count in gender_dist.items():
            print(f"   - {gender}: {count:,} ({count/len(df)*100:.1f}%)")
    
    if 'skin_tone' in df.columns:
        skin_tone_dist = df['skin_tone'].value_counts().sort_index()
        print(f"\n🎨 Monk Skin Tone Distribution (1-10):")
        for tone, count in skin_tone_dist.items():
            print(f"   - MST {tone}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Verify image paths
    print(f"\n{'='*100}")
    print(f" IMAGE PATH VERIFICATION")
    print(f"{'='*100}\n")
    
    existing = []
    missing = []
    categories = defaultdict(lambda: {'existing': [], 'missing': []})
    
    for idx, row in df.iterrows():
        img_path = row['image_path']
        category = get_category_from_path(img_path)
        subcategory = get_subcategory_from_path(img_path)
        
        if os.path.exists(img_path):
            existing.append((idx, row, img_path))
            categories[category]['existing'].append((idx, row, img_path, subcategory))
        else:
            missing.append(img_path)
            categories[category]['missing'].append(img_path)
    
    print(f"✅ Images FOUND on disk: {len(existing):,} ({len(existing)/len(df)*100:.1f}%)")
    print(f"❌ Images MISSING: {len(missing):,} ({len(missing)/len(df)*100:.1f}%)")
    
    if missing:
        print(f"\n⚠️  WARNING: {len(missing):,} images referenced in CSV but not found on disk!")
        print(f"\nFirst 20 missing image paths:")
        for i, path in enumerate(missing[:20], 1):
            print(f"   {i:2d}. {path}")
        
        if len(missing) > 20:
            print(f"   ... and {len(missing) - 20:,} more")
    
    # Category breakdown
    print(f"\n{'='*100}")
    print(f" CATEGORY BREAKDOWN")
    print(f"{'='*100}\n")
    
    for cat in ['Real', 'GANs', 'DMs', 'deepfakes']:
        cat_data = categories.get(cat, {'existing': [], 'missing': []})
        total = len(cat_data['existing']) + len(cat_data['missing'])
        found = len(cat_data['existing'])
        
        if total > 0:
            print(f"\n📁 {cat}:")
            print(f"   Total: {total:,}")
            print(f"   Found: {found:,} ({found/total*100:.1f}%)")
            print(f"   Missing: {len(cat_data['missing']):,}")
            
            # Subcategory breakdown for existing images
            if cat_data['existing']:
                subcats = defaultdict(int)
                for _, _, _, subcat in cat_data['existing']:
                    subcats[subcat] += 1
                
                if subcats:
                    print(f"   Subcategories:")
                    for subcat, count in sorted(subcats.items()):
                        print(f"      - {subcat}: {count:,}")
    
    # Sample images for visual verification
    print(f"\n{'='*100}")
    print(f" SAMPLE IMAGES FOR VISUAL VERIFICATION")
    print(f"{'='*100}\n")
    
    if existing:
        print(f"Below are ~10 sample images per category for manual visual inspection.")
        print(f"Please open these images and verify they match their labels:\n")
        
        for cat in ['Real', 'GANs', 'DMs', 'deepfakes']:
            cat_existing = categories.get(cat, {}).get('existing', [])
            
            if cat_existing:
                n_samples = min(10, len(cat_existing))
                samples = random.sample(cat_existing, n_samples)
                
                print(f"\n{'─'*100}")
                print(f"📷 {cat} Category - {n_samples} Samples (out of {len(cat_existing):,} available)")
                print(f"{'─'*100}")
                
                for i, (idx, row, img_path, subcat) in enumerate(samples, 1):
                    label = 'REAL' if row.get('target', -1) == 0 else 'FAKE'
                    gender = row.get('gender', 'N/A')
                    skin_tone = row.get('skin_tone', 'N/A')
                    
                    print(f"\n   Sample {i}:")
                    print(f"      Path      : {img_path}")
                    print(f"      Subcategory: {subcat}")
                    print(f"      Label     : {label} (target={row.get('target', 'N/A')})")
                    print(f"      Gender    : {gender}")
                    print(f"      Skin Tone : MST {skin_tone}")
                    print(f"      CSV Row   : {idx}")
        
        print(f"\n{'='*100}")
        print(f" ACTION ITEMS")
        print(f"{'='*100}\n")
        print(f"✓ To visually verify these samples:")
        print(f"  1. Open each image file in an image viewer")
        print(f"  2. Verify the image matches its category (Real vs Fake)")
        print(f"  3. Check demographic attributes align with expectations")
        print(f"  4. Look for any mislabeled or corrupted images")
        
    else:
        print(f"❌ No images found on disk. Cannot provide samples for verification.")
        print(f"\n📥 IMAGES NEED TO BE DOWNLOADED")
        print(f"\nThe CSV files reference images from the AI-Face FairnessBench dataset.")
        print(f"Expected directory structure:")
        print(f"   images/")
        print(f"   ├── train/")
        print(f"   │   ├── Real/FFHQ/")
        print(f"   │   ├── GANs/AttGAN/")
        print(f"   │   ├── GANs/StyleGAN2/")
        print(f"   │   ├── DMs/StableDiffusion1.5/")
        print(f"   │   ├── DMs/Palette/")
        print(f"   │   └── deepfakes/dfdc/")
        print(f"   └── test/")
        print(f"       └── (same structure as train)")
        print(f"\n📚 Download instructions:")
        print(f"   1. Visit: https://github.com/purdue-m2/ai-face-fairnessbench")
        print(f"   2. Follow download links to get the image dataset")
        print(f"   3. Extract images to maintain the directory structure above")
    
    return {
        'total': len(df),
        'existing': len(existing),
        'missing': len(missing),
        'categories': {cat: len(data['existing']) for cat, data in categories.items()}
    }


if __name__ == "__main__":
    import sys
    
    # Set random seed for reproducible sampling
    random.seed(42)
    
    print("\n" + "="*100)
    print(" DEEPFAKE DATASET VERIFICATION & SAMPLING TOOL")
    print("="*100)
    
    # Check both train and test datasets
    datasets = [
        ('dataset/train_subset_mapped.csv', 'Training Set'),
        ('dataset/test_subset_mapped.csv', 'Test Set')
    ]
    
    results = {}
    for csv_path, name in datasets:
        result = verify_and_sample_dataset(csv_path, name)
        results[name] = result
    
    # Final summary
    print(f"\n{'='*100}")
    print(f" FINAL SUMMARY")
    print(f"{'='*100}\n")
    
    for name, result in results.items():
        if result:
            status = "✅ READY" if result['missing'] == 0 else "❌ INCOMPLETE"
            print(f"{name:20s}: {status}")
            print(f"   Total: {result['total']:,} | Found: {result['existing']:,} | Missing: {result['missing']:,}")
    
    # Check if we can proceed with training
    all_images_present = all(
        result and result['missing'] == 0 
        for result in results.values() 
        if result is not None
    )
    
    if all_images_present:
        print(f"\n✅ All images verified! Dataset is ready for training.")
    else:
        print(f"\n⚠️  Images missing! Download the AI-Face dataset before training.")
        print(f"   See instructions above or in README.md")
    
    print(f"\n{'='*100}\n")
