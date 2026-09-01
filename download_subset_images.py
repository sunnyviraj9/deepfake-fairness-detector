"""
Download and Extract AI-Face Dataset Subset (30,000 train + 5,000 test images)
Based on: https://github.com/purdue-m2/ai-face-fairnessbench

This script downloads the full AI-Face dataset and extracts only the images
specified in your CSV files, avoiding downloading unnecessary data.
"""

import os
import sys
import subprocess
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def check_git_lfs():
    """Check if git-lfs is installed"""
    try:
        result = subprocess.run(['git', 'lfs', 'version'], 
                              capture_output=True, text=True, check=True)
        print(f"✅ Git LFS found: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Git LFS not found!")
        print("\n📥 Install Git LFS:")
        print("   Windows: Download from https://git-lfs.github.com/")
        print("   Or run: winget install Git.GitLFS")
        print("\n   After installation, run: git lfs install")
        return False


def clone_aiface_repo(target_dir="AI-Face-Full"):
    """Clone the AI-Face repository"""
    repo_url = "https://huggingface.co/datasets/Purdue-M2/AI-Face"
    
    if os.path.exists(target_dir):
        print(f"⚠️  Directory {target_dir} already exists.")
        response = input("   Use existing download? (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            print("✅ Using existing download")
            return target_dir
        else:
            print("   Removing old directory...")
            shutil.rmtree(target_dir)
    
    print(f"\n📥 Cloning AI-Face dataset from HuggingFace...")
    print(f"   This will download ~50-100GB. It may take several hours.")
    print(f"   Target directory: {target_dir}\n")
    
    try:
        # Clone without checking out files first (faster)
        subprocess.run([
            'git', 'clone', 
            '--no-checkout',
            repo_url,
            target_dir
        ], check=True)
        
        print("✅ Repository cloned (metadata only)")
        return target_dir
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to clone repository: {e}")
        return None


def extract_needed_images(csv_file, source_root, target_root, split_name):
    """
    Extract only the images specified in the CSV file
    
    Args:
        csv_file: Path to CSV with image_path column
        source_root: Root directory of full dataset
        target_root: Where to save extracted subset
        split_name: "train" or "test"
    """
    print(f"\n{'='*80}")
    print(f" EXTRACTING {split_name.upper()} IMAGES")
    print(f"{'='*80}\n")
    
    # Load CSV
    if not os.path.exists(csv_file):
        print(f"❌ CSV not found: {csv_file}")
        return 0
    
    df = pd.read_csv(csv_file)
    print(f"📋 CSV loaded: {len(df):,} images to extract")
    
    # Stats
    extracted = 0
    missing = []
    skipped = 0
    
    # Create progress bar
    print(f"\n🔍 Extracting images...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{split_name}"):
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
    print(f"📊 Total: {len(df):,}")
    
    if missing:
        print(f"\n⚠️  {len(missing)} images not found in source dataset")
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


def download_and_extract_subset():
    """Main function to download and extract subset"""
    
    print("\n" + "="*80)
    print(" AI-FACE DATASET SUBSET DOWNLOADER")
    print("="*80)
    print("\nThis script will:")
    print("1. Clone the AI-Face repository (~50-100GB)")
    print("2. Extract only your 35,000 subset images (~5-10GB)")
    print("3. Organize into images/train/ and images/test/")
    print("\n" + "="*80 + "\n")
    
    # Configuration
    train_csv = "dataset/train_subset_mapped_fixed.csv"
    test_csv = "dataset/test_subset_mapped_fixed.csv"
    full_dataset_dir = "AI-Face-Full"
    subset_dir = "."  # Current directory (will create images/ folder)
    
    # Check prerequisites
    print("🔍 Checking prerequisites...\n")
    
    if not check_git_lfs():
        print("\n❌ Git LFS is required. Install it and run 'git lfs install' first.")
        return
    
    # Check CSV files exist
    if not os.path.exists(train_csv):
        print(f"❌ Training CSV not found: {train_csv}")
        return
    
    if not os.path.exists(test_csv):
        print(f"❌ Test CSV not found: {test_csv}")
        return
    
    print(f"✅ Training CSV found: {train_csv}")
    print(f"✅ Test CSV found: {test_csv}")
    
    # Ask for confirmation
    print("\n" + "="*80)
    print(" DOWNLOAD CONFIRMATION")
    print("="*80)
    print("\n⚠️  This will download ~50-100GB of data.")
    print("   Make sure you have:")
    print("   - Sufficient disk space (at least 100GB free)")
    print("   - Stable internet connection")
    print("   - Several hours available (download time varies)")
    
    response = input("\n   Continue? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("❌ Cancelled.")
        return
    
    # Step 1: Clone repository
    print("\n" + "="*80)
    print(" STEP 1: CLONE AI-FACE REPOSITORY")
    print("="*80 + "\n")
    
    repo_dir = clone_aiface_repo(full_dataset_dir)
    if not repo_dir:
        print("❌ Failed to clone repository. Aborting.")
        return
    
    # Step 2: Checkout files (download actual images)
    print("\n" + "="*80)
    print(" STEP 2: DOWNLOAD IMAGE FILES")
    print("="*80 + "\n")
    
    print("📥 Downloading image files with Git LFS...")
    print("   This may take several hours depending on your connection.\n")
    
    try:
        subprocess.run(
            ['git', 'lfs', 'pull'],
            cwd=repo_dir,
            check=True
        )
        print("\n✅ Images downloaded successfully")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to download images: {e}")
        print("   You may need to manually run: cd AI-Face-Full && git lfs pull")
        return
    
    # Step 3: Extract subset
    print("\n" + "="*80)
    print(" STEP 3: EXTRACT SUBSET IMAGES")
    print("="*80 + "\n")
    
    # Extract train images
    train_extracted = extract_needed_images(
        csv_file=train_csv,
        source_root=repo_dir,
        target_root=subset_dir,
        split_name="train"
    )
    
    # Extract test images
    test_extracted = extract_needed_images(
        csv_file=test_csv,
        source_root=repo_dir,
        target_root=subset_dir,
        split_name="test"
    )
    
    # Final summary
    print("\n" + "="*80)
    print(" FINAL SUMMARY")
    print("="*80)
    print(f"\n✅ Total images extracted: {train_extracted + test_extracted:,}")
    print(f"   - Training: {train_extracted:,}")
    print(f"   - Test: {test_extracted:,}")
    
    print(f"\n📁 Images saved to:")
    print(f"   - {os.path.abspath('images/train/')}")
    print(f"   - {os.path.abspath('images/test/')}")
    
    print(f"\n💾 Disk space saved:")
    print(f"   - Full dataset: ~100GB")
    print(f"   - Your subset: ~10GB")
    print(f"   - Saved: ~90GB")
    
    # Cleanup option
    print("\n" + "="*80)
    print(" CLEANUP")
    print("="*80)
    print(f"\n🗑️  The full dataset is still in: {full_dataset_dir}/")
    print(f"   You can delete it to save disk space (~100GB)")
    
    response = input("\n   Delete full dataset now? (yes/no): ").strip().lower()
    if response in ['yes', 'y']:
        print(f"\n   Removing {full_dataset_dir}...")
        shutil.rmtree(full_dataset_dir)
        print("   ✅ Cleanup complete!")
    else:
        print("   Keeping full dataset for future use.")
    
    # Verification
    print("\n" + "="*80)
    print(" NEXT STEPS")
    print("="*80)
    print("\n1. Verify images:")
    print("   python verify_fixed_csvs.py")
    
    print("\n2. Test your pipeline:")
    print("   python train.py --annotations dataset/train_subset_mapped_fixed.csv --epochs 1")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    try:
        download_and_extract_subset()
    except KeyboardInterrupt:
        print("\n\n❌ Download interrupted by user.")
        print("   You can resume by running this script again.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
