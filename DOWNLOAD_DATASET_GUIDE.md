# Complete Guide: Download AI-Face Dataset for Your 30K Subset

This guide shows you how to get only the 35,000 images you need (30k train + 5k test) from the full AI-Face dataset.

---

## 🎯 Goal

Download and extract **only** the 35,000 images specified in your CSV files, instead of the full 1.6M+ images dataset.

**Disk Space:**
- Full dataset: ~100GB  
- Your subset: ~10GB  
- **You save: ~90GB**

---

## 📋 Two Methods

### Method 1: Automatic Download + Extract (Recommended) ⚡
Automatically downloads and extracts only your subset.

### Method 2: Manual Download + Extract
Download full dataset manually, then extract your subset.

---

## ⚡ Method 1: Automatic Download + Extract

This script handles everything automatically.

### Prerequisites:

1. **Install Git LFS:**
   ```bash
   # Windows
   winget install Git.GitLFS
   
   # Or download from: https://git-lfs.github.com/
   
   # After installation, run:
   git lfs install
   ```

2. **Install Python dependencies:**
   ```bash
   pip install tqdm pandas
   ```

### Run the Script:

```bash
python download_subset_images.py
```

**What it does:**
1. ✅ Checks prerequisites (Git LFS)
2. ✅ Clones AI-Face repository from HuggingFace
3. ✅ Downloads all images (~50-100GB)
4. ✅ Extracts only your 35,000 subset images
5. ✅ Organizes into `images/train/` and `images/test/`
6. ✅ Optionally deletes full dataset to save space

**Time:** 2-6 hours depending on internet speed

---

## 📥 Method 2: Manual Download + Extract

### Step 1: Download Full Dataset

**Option A: Git Clone (Recommended)**
```bash
# Install Git LFS first
git lfs install

# Clone repository
git clone https://huggingface.co/datasets/Purdue-M2/AI-Face AI-Face-Full

# Download images
cd AI-Face-Full
git lfs pull
```

**Option B: Direct Download**
Visit: https://huggingface.co/datasets/Purdue-M2/AI-Face/tree/main

Download these archives:
- `Real.tar.gz` (FFHQ, IMDB-WIKI)
- `GANs.tar.gz` (AttGAN, StyleGAN2, etc.)
- `DMs.tar.gz` (Stable Diffusion, Palette, etc.)
- `deepfakes.tar.gz` (DFDC, DFD, FF++, etc.)

Extract all to a single directory: `AI-Face-Full/`

### Step 2: Extract Your Subset

```bash
python extract_subset_from_full.py
```

When prompted, enter the path to your full dataset:
```
Enter path to full dataset: ./AI-Face-Full
```

**What it does:**
1. ✅ Reads your CSV files
2. ✅ Copies only the 35,000 images you need
3. ✅ Organizes into `images/train/` and `images/test/`
4. ✅ Shows progress bar

**Time:** 10-30 minutes (copying files)

### Step 3: Cleanup (Optional)

After extraction, delete the full dataset to save space:
```bash
rm -rf AI-Face-Full  # Saves ~90GB
```

---

## 🔍 Verify Download

After either method, verify images were extracted:

```bash
python verify_fixed_csvs.py
```

**Expected output:**
```
✅ Training Set: 30,000/30,000 images found
✅ Test Set: 5,000/5,000 images found
```

The script will also show 10 sample images per category for visual verification.

---

## 📁 Expected Directory Structure

After extraction:

```
your-project/
├── images/                          ✅ Created by scripts
│   ├── train/
│   │   ├── Real/
│   │   │   ├── FFHQ/
│   │   │   │   ├── face_000000_mst1_female.jpg
│   │   │   │   ├── face_000002_mst3_male.jpg
│   │   │   │   └── ...
│   │   │   └── imdb_wiki/
│   │   ├── GANs/
│   │   │   ├── AttGAN/
│   │   │   │   ├── face_000001_mst2_female.jpg
│   │   │   │   └── ...
│   │   │   └── StyleGAN2/
│   │   ├── DMs/
│   │   │   ├── StableDiffusion1.5/
│   │   │   └── Palette/
│   │   └── deepfakes/
│   │       ├── dfdc/
│   │       └── dfd/
│   └── test/
│       └── (same structure)
├── dataset/
│   ├── train_subset_mapped_fixed.csv  ✅ You have this
│   └── test_subset_mapped_fixed.csv   ✅ You have this
└── ...
```

---

## ⚙️ Scripts Created

### 1. `download_subset_images.py` (All-in-one)
- Clones AI-Face repo
- Downloads images
- Extracts subset
- Cleans up

**Use when:** Starting from scratch

### 2. `extract_subset_from_full.py` (Extract only)
- Extracts subset from pre-downloaded dataset
- Interactive prompts
- Progress bars

**Use when:** Already have full dataset downloaded

### 3. `verify_fixed_csvs.py` (Verification)
- Checks all images exist
- Samples 10 per category
- Shows metadata

**Use after:** Download completes

---

## 🚀 Quick Start Commands

**Easiest (automatic):**
```bash
# Install prerequisites
git lfs install
pip install tqdm pandas

# Download and extract
python download_subset_images.py

# Verify
python verify_fixed_csvs.py

# Start training
python train.py --annotations dataset/train_subset_mapped_fixed.csv --epochs 5
```

**If you already have the full dataset:**
```bash
# Extract subset
python extract_subset_from_full.py

# Verify
python verify_fixed_csvs.py

# Start training
python train.py --annotations dataset/train_subset_mapped_fixed.csv --epochs 5
```

---

## 📊 Dataset Breakdown

Your 35,000 image subset includes:

### Training Set (30,000 images):
- **Real:** 15,000 (50%)
  - FFHQ
  - IMDB-WIKI
- **Fake:** 15,000 (50%)
  - GANs: 6,000 (AttGAN, StyleGAN2)
  - DMs: 6,000 (Stable Diffusion, Palette)
  - Deepfakes: 3,000 (DFDC, DFD)

### Test Set (5,000 images):
- **Real:** 2,500 (50%)
- **Fake:** 2,500 (50%)
  - Same distribution as training

### Demographics (Perfectly Balanced):
- **Gender:** 50% Male, 50% Female
- **Monk Skin Tone:** Equal across MST 1-10
- **Age Groups:** Child, Youth, Adult, Middle-aged, Senior
- **Intersections:** All 6 groups (Male/Female × Light/Medium/Dark)

---

## ⏱️ Time Estimates

| Step | Method 1 (Auto) | Method 2 (Manual) |
|------|-----------------|-------------------|
| Download | 2-6 hours | 2-6 hours |
| Extract | Automatic | 10-30 minutes |
| Verify | 1 minute | 1 minute |
| **Total** | **2-6 hours** | **2.5-6.5 hours** |

*Time varies based on internet speed and disk I/O*

---

## 🆘 Troubleshooting

### Error: "git-lfs not found"
```bash
# Install Git LFS
winget install Git.GitLFS

# Or download: https://git-lfs.github.com/

# Initialize
git lfs install
```

### Error: "No space left on device"
- Need at least 100GB free space during download
- After extraction (10GB), can delete full dataset to recover space

### Error: "Permission denied"
- Run as administrator
- Or choose a directory where you have write permissions

### Images not found in full dataset
- Make sure you downloaded all archives
- Check directory structure matches expected layout
- Try re-downloading missing archives

---

## 💡 Tips

1. **Use fast internet:** Download is 50-100GB
2. **Stable connection:** Use wired connection if possible
3. **Background download:** Can take hours, run overnight
4. **Resume capability:** Git LFS supports resuming interrupted downloads
5. **Disk space:** Keep 100GB free during download, 20GB after cleanup

---

## 📧 Need Help?

**AI-Face Dataset Issues:**
- Email: lin1785@purdue.edu, hu968@purdue.edu
- GitHub: https://github.com/purdue-m2/ai-face-fairnessbench

**Script Issues:**
Check error messages and verify:
- Git LFS is installed
- CSV files exist
- Sufficient disk space
- Internet connection stable

---

## ✅ Success Checklist

- [ ] Git LFS installed and initialized
- [ ] Python dependencies installed (tqdm, pandas)
- [ ] Downloaded/cloned AI-Face dataset
- [ ] Extracted 35,000 subset images
- [ ] Verified with `verify_fixed_csvs.py`
- [ ] All 35,000 images found
- [ ] Directory structure correct
- [ ] (Optional) Deleted full dataset to save space
- [ ] Ready to train!

---

**Ready to download?** Choose your method and get started! 🚀
