"""
AI-Face Dataset Downloader

Based on official repository: https://github.com/purdue-m2/ai-face-fairnessbench
Dataset website: https://sites.google.com/view/aifacedetection/getting-started
"""

import os
import urllib.request
import sys
from pathlib import Path

# Official download links from AI-Face repository
DOWNLOAD_LINKS = {
    "images": "https://huggingface.co/datasets/Purdue-M2/AI-Face/tree/main",
    "annotations": "Requires Google Form submission - see instructions below"
}

def print_download_instructions():
    """Print detailed download instructions"""
    
    print("\n" + "="*80)
    print(" AI-FACE DATASET DOWNLOAD INSTRUCTIONS")
    print("="*80 + "\n")
    
    print("📚 DATASET INFORMATION:")
    print("   - Total Images: 1,646,545")
    print("   - AI-Generated: 1,245,660 images (37 methods)")
    print("   - Real Images: 400,885")
    print("   - Size: ~50-100GB (compressed)")
    print("   - License: CC BY-NC-ND 4.0")
    
    print("\n" + "="*80)
    print(" STEP 1: DOWNLOAD IMAGES")
    print("="*80 + "\n")
    
    print("📥 Option A: HuggingFace (Recommended)")
    print("   URL: https://huggingface.co/datasets/Purdue-M2/AI-Face")
    print("\n   Commands:")
    print("   1. Install git-lfs:")
    print("      git lfs install")
    print("\n   2. Clone the dataset:")
    print("      git clone https://huggingface.co/datasets/Purdue-M2/AI-Face")
    print("\n   3. Or download specific files from:")
    print("      https://huggingface.co/datasets/Purdue-M2/AI-Face/tree/main")
    
    print("\n📥 Option B: Direct Download Links")
    print("   Check the official repository for tar file links:")
    print("   https://github.com/purdue-m2/ai-face-fairnessbench#download")
    
    print("\n   Expected archive files:")
    print("   - Real.tar.gz (FFHQ, IMDB-WIKI)")
    print("   - GANs.tar.gz (AttGAN, StyleGAN2, etc.)")
    print("   - DMs.tar.gz (Stable Diffusion, Palette, etc.)")
    print("   - deepfakes.tar.gz (DFDC, DFD, FF++, etc.)")
    
    print("\n" + "="*80)
    print(" STEP 2: DOWNLOAD ANNOTATIONS")
    print("="*80 + "\n")
    
    print("📋 Annotations require EULA agreement:")
    print("\n   1. Download EULA from:")
    print("      https://github.com/purdue-m2/ai-face-fairnessbench")
    print("\n   2. Sign the EULA document")
    print("\n   3. Fill out Google Form (link in repository)")
    print("      Upload signed EULA")
    print("      Provide participant information")
    print("\n   4. Wait for approval email with download link")
    print("      (Usually within 1-2 business days)")
    print("\n   Note: You already have annotations in dataset/train_subset_mapped_fixed.csv")
    print("         These are sufficient for your current 35k subset!")
    
    print("\n" + "="*80)
    print(" STEP 3: ORGANIZE DIRECTORY STRUCTURE")
    print("="*80 + "\n")
    
    print("📁 After downloading, organize as follows:")
    print("""
   capstone_new/
   ├── images/
   │   ├── train/                    ← Your CSVs point here
   │   │   ├── Real/
   │   │   │   ├── FFHQ/
   │   │   │   └── imdb_wiki/
   │   │   ├── GANs/
   │   │   │   ├── AttGAN/
   │   │   │   ├── StyleGAN2/
   │   │   │   └── ... (10 total)
   │   │   ├── DMs/
   │   │   │   ├── StableDiffusion1.5/
   │   │   │   ├── Palette/
   │   │   │   └── ... (8 total)
   │   │   └── deepfakes/
   │   │       ├── dfdc/
   │   │       ├── dfd/
   │   │       └── ... (video datasets)
   │   └── test/
   │       └── (same structure)
   └── dataset/
       ├── train_subset_mapped_fixed.csv  ✅
       └── test_subset_mapped_fixed.csv   ✅
    """)
    
    print("\n" + "="*80)
    print(" STEP 4: EXTRACT AND VERIFY")
    print("="*80 + "\n")
    
    print("🗜️  Extract downloaded archives:")
    print("   tar -xzf Real.tar.gz -C images/")
    print("   tar -xzf GANs.tar.gz -C images/")
    print("   tar -xzf DMs.tar.gz -C images/")
    print("   tar -xzf deepfakes.tar.gz -C images/")
    
    print("\n   Note: You may need to reorganize into train/ and test/ subdirectories")
    print("         based on your CSV files' image_path column")
    
    print("\n✅ Verify images are correctly placed:")
    print("   python verify_fixed_csvs.py")
    
    print("\n" + "="*80)
    print(" ALTERNATIVE: QUICK START WITH PLACEHOLDERS")
    print("="*80 + "\n")
    
    print("⚡ For immediate code testing (NOT for real training):")
    print("   python generate_placeholder_images.py --mode sample")
    print("\n   This creates 40 test images to verify your pipeline works.")
    
    print("\n" + "="*80)
    print(" CONTACT & SUPPORT")
    print("="*80 + "\n")
    
    print("📧 Questions about dataset access:")
    print("   - lin1785@purdue.edu")
    print("   - hu968@purdue.edu")
    
    print("\n🔗 Official Links:")
    print("   - Repository: https://github.com/purdue-m2/ai-face-fairnessbench")
    print("   - Competition: https://sites.google.com/view/aifacedetection/home")
    print("   - Paper: https://arxiv.org/pdf/2406.00783")
    print("   - HuggingFace: https://huggingface.co/datasets/Purdue-M2/AI-Face")
    
    print("\n" + "="*80)
    print(" ESTIMATED DOWNLOAD TIME")
    print("="*80 + "\n")
    
    print("   Connection Speed  | Download Time")
    print("   ------------------+----------------")
    print("   10 Mbps           | ~11-22 hours")
    print("   50 Mbps           | ~2-4 hours")
    print("   100 Mbps          | ~1-2 hours")
    print("   1 Gbps            | ~7-15 minutes")
    
    print("\n💡 TIP: Use wget or aria2c for resumable downloads:")
    print("   wget -c <download_url>")
    print("   aria2c -x 16 -s 16 <download_url>  # 16 parallel connections")
    
    print("\n" + "="*80 + "\n")


def check_huggingface_cli():
    """Check if HuggingFace CLI is available"""
    try:
        import huggingface_hub
        print("✅ HuggingFace Hub library is installed")
        return True
    except ImportError:
        print("❌ HuggingFace Hub library not found")
        print("\n   Install with: pip install huggingface-hub")
        return False


def attempt_huggingface_download():
    """Attempt to download from HuggingFace"""
    print("\n" + "="*80)
    print(" ATTEMPTING HUGGINGFACE DOWNLOAD")
    print("="*80 + "\n")
    
    if not check_huggingface_cli():
        return False
    
    try:
        from huggingface_hub import snapshot_download
        
        print("🔄 This will download ~50-100GB of data.")
        print("   Make sure you have sufficient disk space and a stable connection.")
        response = input("\nContinue? (yes/no): ").strip().lower()
        
        if response not in ['yes', 'y']:
            print("Cancelled.")
            return False
        
        print("\n📥 Downloading AI-Face dataset from HuggingFace...")
        print("   This may take several hours depending on your connection.\n")
        
        dataset_path = snapshot_download(
            repo_id="Purdue-M2/AI-Face",
            repo_type="dataset",
            local_dir="./AI-Face-downloaded",
            local_dir_use_symlinks=False
        )
        
        print(f"\n✅ Dataset downloaded to: {dataset_path}")
        print("\n📁 Next steps:")
        print("   1. Organize images into images/train/ and images/test/")
        print("   2. Run: python verify_fixed_csvs.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading: {e}")
        print("\n   Try manual download from:")
        print("   https://huggingface.co/datasets/Purdue-M2/AI-Face/tree/main")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI-Face Dataset Download Helper")
    parser.add_argument("--auto", action="store_true", 
                        help="Attempt automatic HuggingFace download")
    parser.add_argument("--instructions-only", action="store_true",
                        help="Only show download instructions (default)")
    
    args = parser.parse_args()
    
    if args.auto:
        attempt_huggingface_download()
    else:
        print_download_instructions()
        
        if not args.instructions_only:
            print("\n" + "="*80)
            print(" AUTOMATIC DOWNLOAD")
            print("="*80 + "\n")
            print("⚡ Want to try automatic download via HuggingFace?")
            print("   This requires huggingface-hub library and ~50-100GB disk space.")
            response = input("\nAttempt automatic download? (yes/no): ").strip().lower()
            
            if response in ['yes', 'y']:
                attempt_huggingface_download()
            else:
                print("\n👍 Follow the manual instructions above to download the dataset.")
                print("\n   Or run: python download_aiface_dataset.py --auto")
