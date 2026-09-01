"""
Verify the FIXED CSV files with enhanced schema
"""
import pandas as pd
import os
from pathlib import Path
from collections import defaultdict
import random

def analyze_fixed_csv(csv_path):
    """Analyze fixed CSV with age and intersection columns"""
    print(f"\n{'='*100}")
    print(f" FIXED CSV ANALYSIS: {csv_path}")
    print(f"{'='*100}\n")
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    print(f"📊 Total entries: {len(df):,}")
    
    # Schema
    print(f"\n📋 CSV Schema:")
    for col in df.columns:
        print(f"   - {col}: {df[col].dtype}")
    
    # Quick stats
    print(f"\n📈 Data Distribution:")
    print(f"   Target: Real={len(df[df['target']==0]):,} | Fake={len(df[df['target']==1]):,}")
    print(f"   Gender: Female={len(df[df['gender']=='Female']):,} | Male={len(df[df['gender']=='Male']):,}")
    print(f"   Skin Tones: {df['skin_tone'].min()} to {df['skin_tone'].max()}")
    print(f"   Age Groups: {sorted(df['age'].unique())}")
    print(f"   Intersection Groups: {sorted(df['intersection'].unique())}")
    
    # Check images
    print(f"\n🔍 Checking image paths...")
    existing = []
    missing = []
    
    for idx, row in df.iterrows():
        if os.path.exists(row['image_path']):
            existing.append(row)
        else:
            missing.append(row['image_path'])
    
    print(f"   Found: {len(existing):,} ({len(existing)/len(df)*100:.1f}%)")
    print(f"   Missing: {len(missing):,} ({len(missing)/len(df)*100:.1f}%)")
    
    if missing:
        print(f"\n   First 10 missing paths:")
        for path in missing[:10]:
            print(f"      {path}")
    
    # Sample by category if images exist
    if existing:
        print(f"\n{'='*100}")
        print(f" SAMPLE IMAGES FOR VISUAL VERIFICATION")
        print(f"{'='*100}\n")
        
        # Group by category
        categories = {
            'Real': [],
            'GANs': [],
            'DMs': [],
            'deepfakes': []
        }
        
        for row in existing:
            path = row['image_path']
            if '/Real/' in path:
                categories['Real'].append(row)
            elif '/GANs/' in path:
                categories['GANs'].append(row)
            elif '/DMs/' in path:
                categories['DMs'].append(row)
            elif '/deepfakes/' in path:
                categories['deepfakes'].append(row)
        
        # Sample 10 per category
        for cat_name, items in categories.items():
            if items:
                n_samples = min(10, len(items))
                samples = random.sample(items, n_samples)
                
                print(f"\n{'─'*100}")
                print(f"📷 {cat_name} - {n_samples} samples")
                print(f"{'─'*100}")
                
                for i, row in enumerate(samples, 1):
                    label = 'REAL' if row['target'] == 0 else 'FAKE'
                    
                    # Get subcategory
                    parts = row['image_path'].split('/')
                    subcat = parts[-2] if len(parts) >= 2 else 'Unknown'
                    
                    print(f"\n   Sample {i}:")
                    print(f"      📁 Path      : {row['image_path']}")
                    print(f"      🏷️  Label     : {label}")
                    print(f"      📂 Subcategory: {subcat}")
                    print(f"      👤 Gender    : {row['gender']}")
                    print(f"      🎨 Skin Tone : MST {row['skin_tone']}")
                    print(f"      🎂 Age Group : {row['age']}")
                    print(f"      🔗 Intersection: {row['intersection']}")
                
                print(f"\n   ✓ Open these {n_samples} images to verify they match labels")
    
    return {
        'total': len(df),
        'existing': len(existing),
        'missing': len(missing)
    }

if __name__ == "__main__":
    random.seed(42)
    
    print("\n" + "="*100)
    print(" FIXED CSV VERIFICATION TOOL")
    print("="*100)
    
    # Analyze both fixed CSVs
    train_result = analyze_fixed_csv('dataset/train_subset_mapped_fixed.csv')
    test_result = analyze_fixed_csv('dataset/test_subset_mapped_fixed.csv')
    
    # Summary
    print(f"\n{'='*100}")
    print(f" SUMMARY")
    print(f"{'='*100}\n")
    
    if train_result:
        status = "✅" if train_result['missing'] == 0 else "⚠️"
        print(f"{status} Training Set: {train_result['existing']:,}/{train_result['total']:,} images found")
    
    if test_result:
        status = "✅" if test_result['missing'] == 0 else "⚠️"
        print(f"{status} Test Set: {test_result['existing']:,}/{test_result['total']:,} images found")
    
    if train_result and test_result:
        all_found = (train_result['missing'] == 0 and test_result['missing'] == 0)
        if all_found:
            print(f"\n✅ All images found! Ready for visual verification.")
            print(f"   Review the samples above and confirm labels match images.")
        else:
            print(f"\n⚠️  Some images missing. Download AI-Face dataset to proceed.")
    
    print(f"\n{'='*100}\n")
