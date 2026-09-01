"""
Extract Subset Images from Full AI-Face Dataset
Use this if you've already downloaded the full dataset archives
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def extract_subset_images(csv_file, source_root, target_root, split_name):
    """
    Extract images specified in CSV from full dataset
    
    Args:
        csv_file: Path to CSV with image_path column
        source_root: Root directory of full downloaded dataset
        target_root: Where to save extracted subset
        split_name: "train" or "test"
    """
    print(f"\n{'='*80}")
    print(f" EXTRACTING {split_name.upper()} SUBSET")
    print(f"{'='*80}\n")
    
    # Check CSV exists
    if not os.path.exists(csv_file):
        print(f"❌ CSV not found: {csv_file}")
        return 0
    
    # Load CSV
    df = pd.read_csv(csv_file)
    print(f"📋 CSV loaded: {len(df):,} images to extract")
    print(f"📂 Source: {source_root}")
    print(f"📁 Target: {target_root}")
    
    # Check source directory exists
    if not os.path.exists(source_root):
        print(f"\n❌ Source directory not found: {source_root}")
        print(f"   Make sure you've downloaded and extracted the full AI-Face dataset")
        return 0
    
    # Stats
    extracted = 0
    missing = []
    skipped = 0
    
    # Extract images
    print(f"\n🔍 Extracting images...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=split_name):
        rel_path = row['image_path']
        
        # Source file in full dataset
        src_file = os.path.join(source_root, rel_path)
        
        # Destination file in subset
        dst_file = os.path.join(target_root, rel_path)
        
        # Skip if already exists
        if os.path.exists(dst_file):
            skipped += 1
            continue
        
        # Check if source exists
        if not os.path.exists(src_file):
            missing.append(rel_path)
            continue
        
        # Create directory structure
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        
        # Copy file
        try:
            shutil.copy2(src_file, dst_file)
            extracted += 1
        except Exception as e:
            print(f"\n⚠️  Error copying {rel_path}: {e}")
            missing.append(rel_path)
    
    # Summary
    print(f"\n{'='*80}")
    print(f" {split_name.upper()} EXTRACTION SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Extracted: {extracted:,}")
    print(f"⏭️  Skipped (already exist): {skipped:,}")
    print(f"❌ Missing: {len(missing):,}")
    print(f"📊 Total in CSV: {len(df):,}")
    
    if missing:
        print(f"\n⚠️  {len(missing)} images not found in source")
        if len(missing) <= 10:
            print("   Missing files:")
            for path in missing:
                print(f"      - {path}")
        else:
            print(f"   First 10 missing files:")
            for path in missing[:10]:
                print(f"      - {path}")
            print(f"   ... and {len(missing) - 10} more")
    
    return extracted


def main():
    """Main extraction function"""
    
    print("\n" + "="*80)
    print(" AI-FACE SUBSET EXTRACTOR")
    print("="*80)
    print("\nExtracts only the images needed for your 30k train + 5k test subset")
    print("from the full AI-Face dataset.\n")
    print("="*80 + "\n")
    
    # Configuration
    train_csv = "dataset/train_subset_mapped_fixed.csv"
    test_csv = "dataset/test_subset_mapped_fixed.csv"
    
    # Ask for source directory
    print("📂 Where is your full AI-Face dataset located?")
    print("   Examples:")
    print("   - ./AI-Face-Full")
    print("   - C:/datasets/AI-Face")
    print("   - /mnt/data/AI-Face")
    
    source_root = input("\nEnter path to full dataset: ").strip()
    
    if not source_root:
        source_root = "./AI-Face-Full"
        print(f"   Using default: {source_root}")
    
    # Remove quotes if user pasted path with quotes
    source_root = source_root.strip('"').strip("'")
    
    # Check if source exists
    if not os.path.exists(source_root):
        print(f"\n❌ Directory not found: {source_root}")
        print("\n💡 Download the full dataset first:")
        print("   python download_subset_images.py")
        return
    
    print(f"\n✅ Source directory found: {source_root}")
    
    # Target directory (current directory, will create images/ folder)
    target_root = "."
    
    # Check CSV files
    if not os.path.exists(train_csv):
        print(f"\n❌ Training CSV not found: {train_csv}")
        return
    
    if not os.path.exists(test_csv):
        print(f"\n❌ Test CSV not found: {test_csv}")
        return
    
    print(f"✅ Training CSV found: {train_csv}")
    print(f"✅ Test CSV found: {test_csv}")
    
    # Confirmation
    print("\n" + "="*80)
    response = input("Continue with extraction? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("❌ Cancelled.")
        return
    
    # Extract train images
    train_extracted = extract_subset_images(
        csv_file=train_csv,
        source_root=source_root,
        target_root=target_root,
        split_name="train"
    )
    
    # Extract test images
    test_extracted = extract_subset_images(
        csv_file=test_csv,
        source_root=source_root,
        target_root=target_root,
        split_name="test"
    )
    
    # Final summary
    print("\n" + "="*80)
    print(" EXTRACTION COMPLETE")
    print("="*80)
    print(f"\n✅ Total images extracted: {train_extracted + test_extracted:,}")
    print(f"   - Training: {train_extracted:,} / 30,000")
    print(f"   - Test: {test_extracted:,} / 5,000")
    
    print(f"\n📁 Images saved to:")
    print(f"   - {os.path.abspath('images/train/')}")
    print(f"   - {os.path.abspath('images/test/')}")
    
    # Verification
    print("\n" + "="*80)
    print(" NEXT STEPS")
    print("="*80)
    print("\n1. Verify images were extracted correctly:")
    print("   python verify_fixed_csvs.py")
    
    print("\n2. If all images found, you can delete the full dataset to save space")
    print(f"   rm -rf {source_root}  # Saves ~90GB")
    
    print("\n3. Start training:")
    print("   python train.py --annotations dataset/train_subset_mapped_fixed.csv")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Extraction interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
