"""
Cleanup Redundant Files Script
Identifies and removes duplicate/obsolete documentation and verification files
"""

import os
import shutil

# Files to delete (redundant/obsolete)
REDUNDANT_FILES = [
    # Redundant verification scripts (keep only verify_fixed_csvs.py)
    "verify_dataset.py",                    # Obsolete, replaced by verify_fixed_csvs.py
    "verify_and_sample_dataset.py",        # Obsolete, replaced by verify_fixed_csvs.py
    
    # Redundant download script (keep only download_subset_images.py)
    "download_aiface_dataset.py",           # Obsolete, replaced by download_subset_images.py
    
    # Redundant documentation (keep only essential guides)
    "DATASET_VERIFICATION_REPORT.md",       # Obsolete initial report
    "FIXED_CSV_REPORT.md",                  # Obsolete, info covered in DOWNLOAD_DATASET_GUIDE.md
    "HOW_TO_FIX_MISSING_IMAGES.md",        # Obsolete, covered in DOWNLOAD_DATASET_GUIDE.md
    "GITHUB_SETUP_INSTRUCTIONS.md",         # Obsolete, already pushed to GitHub
    "PUSH_TO_GITHUB.md",                    # Obsolete, already pushed to GitHub
    "REPOSITORY_LIVE.md",                   # Obsolete, repository is live
]

# Files to keep (essential)
ESSENTIAL_FILES = [
    "README.md",                            # Main project documentation
    "DOWNLOAD_DATASET_GUIDE.md",           # Dataset download instructions
    "requirements.txt",                     # Python dependencies
    "train.py",                             # Training script
    "evaluate.py",                          # Evaluation script
    "inference.py",                         # Inference script
    "benchmark.py",                         # Benchmarking script
    "download_subset_images.py",           # Main download script
    "extract_subset_from_full.py",         # Extraction helper
    "verify_fixed_csvs.py",                # Main verification script
    "generate_placeholder_images.py",      # Placeholder generation for testing
    ".gitignore",                          # Git configuration
]


def cleanup_files(dry_run=True):
    """
    Delete redundant files
    
    Args:
        dry_run: If True, only show what would be deleted without deleting
    """
    print("\n" + "="*80)
    print(" CLEANUP REDUNDANT FILES")
    print("="*80 + "\n")
    
    if dry_run:
        print("🔍 DRY RUN MODE - No files will be deleted")
        print("   Run with --confirm to actually delete files\n")
    else:
        print("⚠️  DELETION MODE - Files will be permanently deleted\n")
    
    deleted = []
    not_found = []
    
    print("📋 Files to delete:\n")
    
    for file in REDUNDANT_FILES:
        if os.path.exists(file):
            size = os.path.getsize(file)
            size_kb = size / 1024
            
            if dry_run:
                print(f"   🗑️  Would delete: {file} ({size_kb:.1f} KB)")
            else:
                try:
                    os.remove(file)
                    print(f"   ✅ Deleted: {file} ({size_kb:.1f} KB)")
                    deleted.append(file)
                except Exception as e:
                    print(f"   ❌ Error deleting {file}: {e}")
        else:
            not_found.append(file)
    
    if not_found:
        print(f"\n⏭️  Already deleted: {len(not_found)} files")
    
    # Summary
    print("\n" + "="*80)
    print(" SUMMARY")
    print("="*80 + "\n")
    
    if dry_run:
        print(f"📊 Would delete: {len(REDUNDANT_FILES) - len(not_found)} files")
        print(f"⏭️  Already gone: {len(not_found)} files")
        print(f"✅ Will keep: {len(ESSENTIAL_FILES)} essential files")
        print(f"\n💡 Run again with --confirm to actually delete")
    else:
        print(f"✅ Deleted: {len(deleted)} files")
        print(f"⏭️  Already gone: {len(not_found)} files")
        print(f"✅ Kept: {len(ESSENTIAL_FILES)} essential files")
    
    print("\n" + "="*80)
    print(" ESSENTIAL FILES KEPT")
    print("="*80 + "\n")
    
    for file in ESSENTIAL_FILES:
        if os.path.exists(file):
            print(f"   ✅ {file}")
    
    print("\n" + "="*80 + "\n")
    
    return deleted


def show_project_structure():
    """Show the clean project structure after cleanup"""
    print("\n" + "="*80)
    print(" CLEAN PROJECT STRUCTURE")
    print("="*80 + "\n")
    
    structure = """
    capstone_new/
    ├── models/                      # Model architectures
    │   ├── spatial_detector.py
    │   ├── frequency_detector.py
    │   └── dual_stream_detector.py
    ├── utils/                       # Utilities
    │   ├── dataset_loader.py
    │   ├── metrics.py
    │   └── loss.py
    ├── data/                        # Data generation scripts
    │   ├── generate_mock_data.py
    │   └── mock_dataset/
    ├── dataset/                     # CSV annotations
    │   ├── train_subset_mapped_fixed.csv
    │   └── test_subset_mapped_fixed.csv
    ├── tests/                       # Unit tests
    ├── configs/                     # Configuration files
    ├── train.py                     # Training script
    ├── evaluate.py                  # Evaluation script
    ├── inference.py                 # Inference script
    ├── benchmark.py                 # Benchmarking script
    ├── download_subset_images.py   # Download dataset
    ├── extract_subset_from_full.py # Extract subset
    ├── verify_fixed_csvs.py        # Verify dataset
    ├── generate_placeholder_images.py # Test images
    ├── README.md                    # Main documentation
    ├── DOWNLOAD_DATASET_GUIDE.md   # Download guide
    ├── requirements.txt             # Dependencies
    └── .gitignore                   # Git config
    """
    
    print(structure)
    print("="*80 + "\n")


if __name__ == "__main__":
    import sys
    
    # Check for --confirm flag
    confirm = "--confirm" in sys.argv
    
    if not confirm:
        print("\n⚠️  This script will delete redundant/obsolete files")
        print("   Run in dry-run mode first to see what will be deleted\n")
    
    # Run cleanup
    deleted = cleanup_files(dry_run=not confirm)
    
    # Show clean structure
    if confirm and deleted:
        show_project_structure()
        
        print("🎉 Cleanup complete!")
        print("\n📝 Next steps:")
        print("   1. Commit changes: git add -A")
        print("   2. Commit: git commit -m 'Remove redundant files'")
        print("   3. Push: git push origin main")
    elif not confirm:
        print("\n💡 To actually delete files, run:")
        print("   python cleanup_redundant_files.py --confirm")
