# How to Fix Missing Images

You have 2 options depending on your goal:

---

## ✅ Option 1: Generate Placeholder Images (For Testing Code)

**Use this if:** You want to test your training/evaluation pipeline quickly without downloading 3-5GB of data.

**⚠️ WARNING:** These are synthetic placeholders, NOT real deepfake data. DO NOT use for actual model training/evaluation.

### Quick Test (40 images):
```bash
python generate_placeholder_images.py --mode sample
```

This creates:
- `sample_images/train/` - 40 sample training images
- `sample_images/test/` - 40 sample test images

Then test your code:
```bash
# Update your code to point to sample_images/ instead of images/
python train.py --annotations dataset/train_subset_mapped_fixed.csv --epochs 1
```

### Generate All 35,000 Placeholders:
```bash
python generate_placeholder_images.py --mode full
```

**Note:** Takes 5-10 minutes, creates ~500MB of placeholder images.

### Custom Amount:
```bash
# Generate first 1000 images from each CSV
python generate_placeholder_images.py --mode custom --num_samples 1000
```

---

## ✅ Option 2: Download Real AI-Face Dataset (For Production)

**Use this if:** You want to train a real deepfake detector with actual data.

### Step 1: Visit Official Repository
```
https://github.com/purdue-m2/ai-face-fairnessbench
```

### Step 2: Find Dataset Downloads
Look for:
- Download links (usually OneDrive/SharePoint from Purdue)
- Dataset size: ~3-5GB compressed, 600k+ images
- May require registration or request access

### Step 3: Download & Extract

**What you're looking for:**
- Training images (Real, GANs, DMs, deepfakes)
- Test images
- Annotations (you already have these in your CSVs)

### Step 4: Organize Directory Structure

After downloading, your directory should look like:

```
capstone_new/
├── images/                          ← Create this folder
│   ├── train/
│   │   ├── Real/
│   │   │   ├── FFHQ/               ← Real faces
│   │   │   │   ├── face_000000_mst1_female.jpg
│   │   │   │   ├── face_000002_mst3_male.jpg
│   │   │   │   └── ...
│   │   │   └── imdb_wiki/
│   │   ├── GANs/
│   │   │   ├── AttGAN/             ← GAN-generated
│   │   │   │   ├── face_000001_mst2_female.jpg
│   │   │   │   └── ...
│   │   │   └── StyleGAN2/
│   │   ├── DMs/                    ← Diffusion models
│   │   │   ├── StableDiffusion1.5/
│   │   │   └── Palette/
│   │   └── deepfakes/
│   │       └── dfdc/               ← Deepfake videos
│   └── test/
│       └── (same structure as train)
├── dataset/
│   ├── train_subset_mapped_fixed.csv  ✅ You have this
│   └── test_subset_mapped_fixed.csv   ✅ You have this
└── ...
```

### Step 5: Verify Images

After extraction, run:
```bash
python verify_fixed_csvs.py
```

Should show:
```
✅ Training Set: 30,000/30,000 images found
✅ Test Set: 5,000/5,000 images found
```

---

## What If Download Link Is Not Available?

### Alternative 1: Contact Authors
- Email authors from the paper
- Request access to dataset
- Mention you're using it for research/education

### Alternative 2: Use Component Datasets

The AI-Face dataset is composed of several public datasets. You could download components:

**Real Faces:**
- FFHQ: https://github.com/NVlabs/ffhq-dataset
- IMDB-Wiki: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/

**Fake Faces:**
- DFDC (Deepfake Detection Challenge): https://ai.facebook.com/datasets/dfdc/
- StyleGAN samples: Generate using official StyleGAN2
- Other GAN/diffusion models: Generate samples

Then organize them into the expected structure.

### Alternative 3: Use Mock Dataset

For pure code testing:
```bash
# Use the existing mock dataset (100 images)
python train.py --annotations data/mock_dataset/annotations.csv --epochs 1
```

---

## Quick Decision Tree

```
Do you need to test code NOW?
├─ YES → Use Option 1 (Placeholder Images)
│         python generate_placeholder_images.py --mode sample
│
└─ NO → Need real results?
    └─ YES → Use Option 2 (Download Real Dataset)
             Visit: https://github.com/purdue-m2/ai-face-fairnessbench
```

---

## Verification Commands

After getting images (either option):

```bash
# Check if images exist and get samples for visual inspection
python verify_fixed_csvs.py

# Or use the original tool
python verify_and_sample_dataset.py
```

---

## Summary

| Option | Speed | Quality | Use Case |
|--------|-------|---------|----------|
| **Placeholder Images** | ⚡ Fast (minutes) | 🎨 Synthetic | Code testing |
| **Real Dataset** | 🐌 Slow (download) | ✅ Production | Real training |
| **Mock Dataset** | ⚡ Instant (ready) | 🎨 Limited (100) | Quick tests |

Choose based on your immediate needs!
