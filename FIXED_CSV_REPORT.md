# Fixed CSV Verification Report

**Date:** September 1, 2026  
**Status:** CSV files ready, images missing

---

## Overview

You've added two "fixed" CSV files with enhanced schema that includes additional demographic attributes:
- `dataset/train_subset_mapped_fixed.csv` (30,000 entries)
- `dataset/test_subset_mapped_fixed.csv` (5,000 entries)

---

## CSV Schema Comparison

### Original CSV Schema:
```
- image_path (string)
- target (int: 0=Real, 1=Fake)
- gender (string: Male/Female)
- skin_tone (int: 1-10, Monk Scale)
```

### Fixed CSV Schema (Enhanced):
```
- image_path (string)
- target (int: 0=Real, 1=Fake)  
- gender (string: Male/Female)
- skin_tone (int: 1-10, Monk Scale)
- age (int: 0,1,3,4,5 - Age groups)          ✨ NEW
- intersection (int: 0-5 - Demographic groups) ✨ NEW
```

---

## Fixed CSV Data Quality

### Training Set (`train_subset_mapped_fixed.csv`)

✅ **30,000 entries** perfectly balanced:
- **Target:** 15,000 Real | 15,000 Fake (50/50)
- **Gender:** 15,000 Female | 15,000 Male (50/50)
- **Skin Tone:** Equal distribution across MST 1-10
- **Age Groups:** [0, 1, 3, 4, 5]
- **Intersection Groups:** [0, 1, 2, 3, 4, 5]

### Test Set (`test_subset_mapped_fixed.csv`)

✅ **5,000 entries** perfectly balanced:
- **Target:** 2,500 Real | 2,500 Fake (50/50)
- **Gender:** 2,500 Female | 2,500 Male (50/50)
- **Skin Tone:** Equal distribution across MST 1-10
- **Age Groups:** [0, 1, 3, 4, 5]
- **Intersection Groups:** [0, 1, 2, 3, 4, 5]

---

## Image Path Verification

### Current Status: ❌ **ALL IMAGES MISSING**

- **Training images:** 0 / 30,000 found (0.0%)
- **Test images:** 0 / 5,000 found (0.0%)

Both CSV files reference the same image paths as the original CSVs:
```
images/train/Real/FFHQ/face_000000_mst1_female.jpg
images/train/GANs/AttGAN/face_000001_mst2_female.jpg
images/train/DMs/StableDiffusion1.5/face_000003_mst4_male.jpg
images/train/deepfakes/dfdc/face_000005_mst6_female.jpg
...
```

---

## What "Fixed" Means

The "fixed" CSVs add two important demographic attributes:

### 1. Age Groups (`age` column)
Maps to AI-Face benchmark age categories:
- **0:** Child
- **1:** Youth  
- **3:** Adult
- **4:** Middle-aged
- **5:** Senior

### 2. Intersection Groups (`intersection` column)
Represents **Gender × Skin Tone** intersectional demographics:
- **0:** Female_Light (MST 1-3)
- **1:** Female_Medium (MST 4-7)
- **2:** Female_Dark (MST 8-10)
- **3:** Male_Light (MST 1-3)
- **4:** Male_Medium (MST 4-7)
- **5:** Male_Dark (MST 8-10)

This matches the AI-Face FairnessBench structure for comprehensive bias auditing.

---

## Sample Data (First 10 rows - Training Set)

| image_path | target | gender | skin_tone | age | intersection |
|------------|--------|--------|-----------|-----|--------------|
| images/train/Real/FFHQ/face_000000_mst1_female.jpg | 0 | Female | 1 | 0 | 0 |
| images/train/GANs/AttGAN/face_000001_mst2_female.jpg | 1 | Female | 2 | 1 | 0 |
| images/train/Real/FFHQ/face_000002_mst3_male.jpg | 0 | Male | 3 | 3 | 3 |
| images/train/DMs/StableDiffusion1.5/face_000003_mst4_male.jpg | 1 | Male | 4 | 4 | 4 |
| images/train/Real/FFHQ/face_000004_mst5_female.jpg | 0 | Female | 5 | 5 | 1 |
| images/train/deepfakes/dfdc/face_000005_mst6_female.jpg | 1 | Female | 6 | 0 | 1 |
| images/train/Real/FFHQ/face_000006_mst7_male.jpg | 0 | Male | 7 | 1 | 4 |
| images/train/GANs/StyleGAN2/face_000007_mst8_male.jpg | 1 | Male | 8 | 3 | 5 |
| images/train/Real/FFHQ/face_000008_mst9_female.jpg | 0 | Female | 9 | 4 | 2 |
| images/train/DMs/Palette/face_000009_mst10_female.jpg | 1 | Female | 10 | 5 | 2 |

---

## Visual Verification Plan

### Once Images Are Downloaded:

Run the verification script:
```bash
python verify_fixed_csvs.py
```

The script will:
1. ✅ Confirm all 35,000 image files exist
2. 📊 Show category distributions (Real/GANs/DMs/deepfakes)
3. 🎯 Sample **~10 images per category** for manual inspection
4. 📋 Display full metadata for each sample

### Manual Verification Checklist:

For each of the ~40 sampled images (10 per category):

#### 1. **File Integrity**
- [ ] Image opens without errors
- [ ] No corruption or blank images
- [ ] File size reasonable (20KB - 2MB typical)

#### 2. **Content Verification**
- [ ] Image contains a human face
- [ ] Face is centered and clearly visible
- [ ] Appropriate resolution and quality

#### 3. **Label Accuracy**
Compare CSV metadata with visual inspection:

| Attribute | What to Check |
|-----------|---------------|
| **Target (Real/Fake)** | Real images: natural lighting, realistic textures, no artifacts<br>Fake images: may show GAN/diffusion artifacts, unnatural features |
| **Category** | Matches subcategory (FFHQ, StyleGAN2, etc.) |
| **Gender** | Visual appearance aligns with Male/Female label (when determinable) |
| **Skin Tone** | Rough alignment with Monk Scale 1-10 annotation |
| **Age** | Rough alignment with age group (Child/Youth/Adult/Middle-aged/Senior) |

#### 4. **Category-Specific Checks**

**Real Images (FFHQ, imdb_wiki):**
- Natural photo quality
- Realistic skin texture and pores
- Consistent lighting
- No obvious generation artifacts

**GAN Images (AttGAN, StyleGAN2):**
- May show "GAN artifacts" like checkerboard patterns
- Unusual or synthetic-looking backgrounds
- Overly smooth or plastic-like skin
- Potential issues with teeth, ears, or glasses

**Diffusion Models (Stable Diffusion, Palette):**
- May show diffusion process artifacts
- Overly perfect or "painted" appearance
- Potential blurring or dream-like quality

**Deepfakes (DFDC):**
- Face-swap boundary artifacts
- Color mismatches between face and background
- Temporal inconsistencies (if from video)
- Blending artifacts around jawline

---

## Tools Created

### 1. `verify_fixed_csvs.py`
**Purpose:** Comprehensive verification for the fixed CSV files

**Usage:**
```bash
python verify_fixed_csvs.py
```

**What it does:**
- Analyzes enhanced schema (age, intersection)
- Checks image existence
- Samples 10 images per category for manual review
- Displays full metadata for each sample

### 2. `verify_and_sample_dataset.py`
**Purpose:** Original verification tool (works with both original and fixed CSVs)

**Usage:**
```bash
python verify_and_sample_dataset.py
```

---

## Action Required

### 🔴 IMMEDIATE: Download Images

The CSV files are ready, but all 35,000 images are missing.

**Steps:**
1. Visit: https://github.com/purdue-m2/ai-face-fairnessbench
2. Follow download instructions (likely Purdue SharePoint link)
3. Extract images maintaining directory structure:
   ```
   images/
   ├── train/
   │   ├── Real/FFHQ/
   │   ├── GANs/AttGAN/
   │   ├── GANs/StyleGAN2/
   │   ├── DMs/StableDiffusion1.5/
   │   ├── DMs/Palette/
   │   └── deepfakes/dfdc/
   └── test/
       └── (same structure)
   ```

### 🟢 AFTER Download: Visual Verification

1. Run: `python verify_fixed_csvs.py`
2. Open the ~40 sampled image paths shown in output
3. Verify each image matches its labels using checklist above
4. Document any mismatches or anomalies

---

## Summary

| Aspect | Status |
|--------|--------|
| **CSV Files** | ✅ Ready (enhanced schema) |
| **Data Balance** | ✅ Perfect (50/50 splits) |
| **Schema** | ✅ Enhanced with age + intersection |
| **Images** | ❌ Missing (0/35,000) |
| **Ready for Training** | ❌ No (need images) |

**Next Step:** Download the 35,000 images from AI-Face FairnessBench, then run visual verification.

---

**Report Generated:** September 1, 2026  
**Tool:** `verify_fixed_csvs.py`
