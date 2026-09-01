# Dataset Verification Report

**Date:** September 1, 2026  
**Verified By:** Automated Dataset Verification Tool

---

## Executive Summary

❌ **DATASET STATUS: INCOMPLETE**

All CSV annotation files are properly formatted and contain the expected metadata, but **100% of the referenced image files are missing from disk** (35,000 total images).

---

## CSV Files Verified

### 1. Training Set: `dataset/train_subset_mapped.csv`
- **Total Entries:** 30,000
- **CSV Status:** ✅ Valid and well-formatted
- **Images Found:** 0 (0.0%)
- **Images Missing:** 30,000 (100.0%)

#### Data Distribution
| Attribute | Distribution |
|-----------|-------------|
| **Target** | Real (0): 15,000 (50%) / Fake (1): 15,000 (50%) |
| **Gender** | Female: 15,000 (50%) / Male: 15,000 (50%) |
| **Monk Skin Tone** | Perfectly balanced: 3,000 samples per tone (MST 1-10) |

#### Category Breakdown
| Category | Total | Found | Missing |
|----------|-------|-------|---------|
| **Real** | 15,000 | 0 | 15,000 |
| **GANs** | 6,000 | 0 | 6,000 |
| **DMs** (Diffusion Models) | 6,000 | 0 | 6,000 |
| **deepfakes** | 3,000 | 0 | 3,000 |

### 2. Test Set: `dataset/test_subset_mapped.csv`
- **Total Entries:** 5,000
- **CSV Status:** ✅ Valid and well-formatted
- **Images Found:** 0 (0.0%)
- **Images Missing:** 5,000 (100.0%)

#### Data Distribution
| Attribute | Distribution |
|-----------|-------------|
| **Target** | Real (0): 2,500 (50%) / Fake (1): 2,500 (50%) |
| **Gender** | Female: 2,500 (50%) / Male: 2,500 (50%) |
| **Monk Skin Tone** | Perfectly balanced: 500 samples per tone (MST 1-10) |

#### Category Breakdown
| Category | Total | Found | Missing |
|----------|-------|-------|---------|
| **Real** | 2,500 | 0 | 2,500 |
| **GANs** | 1,000 | 0 | 1,000 |
| **DMs** (Diffusion Models) | 1,000 | 0 | 1,000 |
| **deepfakes** | 500 | 0 | 500 |

---

## Missing Image Paths (Sample)

### Training Set Examples:
```
images/train/Real/FFHQ/face_000000_mst1_female.jpg
images/train/GANs/AttGAN/face_000001_mst2_female.jpg
images/train/Real/FFHQ/face_000002_mst3_male.jpg
images/train/DMs/StableDiffusion1.5/face_000003_mst4_male.jpg
images/train/Real/FFHQ/face_000004_mst5_female.jpg
images/train/deepfakes/dfdc/face_000005_mst6_female.jpg
images/train/Real/FFHQ/face_000006_mst7_male.jpg
images/train/GANs/StyleGAN2/face_000007_mst8_male.jpg
images/train/Real/FFHQ/face_000008_mst9_female.jpg
images/train/DMs/Palette/face_000009_mst10_female.jpg
```

### Test Set Examples:
```
images/test/Real/FFHQ/face_000000_mst1_female.jpg
images/test/GANs/AttGAN/face_000001_mst2_female.jpg
images/test/Real/FFHQ/face_000002_mst3_male.jpg
images/test/DMs/StableDiffusion1.5/face_000003_mst4_male.jpg
images/test/Real/FFHQ/face_000004_mst5_female.jpg
images/test/deepfakes/dfdc/face_000005_mst6_female.jpg
```

---

## Expected Directory Structure

The CSV files expect the following directory structure:

```
capstone_new/
├── images/
│   ├── train/
│   │   ├── Real/
│   │   │   ├── FFHQ/                    (Real faces from FFHQ dataset)
│   │   │   └── imdb_wiki/               (Real faces from IMDB-Wiki)
│   │   ├── GANs/
│   │   │   ├── AttGAN/                  (GAN-generated faces)
│   │   │   └── StyleGAN2/               (StyleGAN2-generated faces)
│   │   ├── DMs/
│   │   │   ├── StableDiffusion1.5/      (Stable Diffusion generated)
│   │   │   └── Palette/                 (Palette diffusion model)
│   │   └── deepfakes/
│   │       └── dfdc/                    (Deepfake Detection Challenge)
│   └── test/
│       └── (same structure as train/)
├── dataset/
│   ├── train_subset_mapped.csv          ✅ EXISTS
│   └── test_subset_mapped.csv           ✅ EXISTS
└── ...
```

---

## Visual Verification Checklist

### ❌ UNABLE TO COMPLETE - Images Not Available

Once images are downloaded, the following verification should be performed:

#### For Each Category (Real/GANs/DMs/deepfakes):
1. **Sample 10 images** from each category
2. **Visual inspection checklist:**
   - [ ] Image opens without corruption
   - [ ] Image contains a face
   - [ ] Face quality matches expected category:
     - **Real:** Natural lighting, realistic textures, no artifacts
     - **GANs:** May show synthetic patterns, unusual skin textures
     - **DMs:** May show diffusion artifacts, overly smooth areas
     - **deepfakes:** Face-swap artifacts, blending issues
   - [ ] Gender label matches visual appearance (when determinable)
   - [ ] Skin tone roughly aligns with Monk Scale annotation
   - [ ] Image metadata (EXIF) doesn't contradict label

3. **Statistical checks:**
   - [ ] No duplicate images across categories
   - [ ] File sizes within reasonable range (20KB - 2MB typical)
   - [ ] Image dimensions consistent (or documented if varying)
   - [ ] No placeholder or error images

---

## Action Items

### 🔴 CRITICAL - Before Training:
1. **Download the AI-Face FairnessBench dataset**
   - Visit: https://github.com/purdue-m2/ai-face-fairnessbench
   - Download images from Purdue SharePoint (link in repository)
   - Extract to maintain expected directory structure

2. **Verify image extraction**
   - Run: `python verify_and_sample_dataset.py`
   - Confirm all 35,000 images are found

3. **Perform visual sampling**
   - Open 10 samples per category as output by the tool
   - Manually verify labels match visual content
   - Document any discrepancies

### 🟡 OPTIONAL - Use Mock Dataset for Testing:
If you want to test the training pipeline without downloading the full dataset:
- Mock dataset available at: `data/mock_dataset/`
- Contains 100 synthetic images with proper annotations
- Run training test: `python train.py --annotations data/mock_dataset/annotations.csv --epochs 1`

---

## CSV Schema Validation

Both CSV files follow the expected schema:

| Column | Type | Values | Valid? |
|--------|------|--------|--------|
| `image_path` | string | Relative paths to images | ✅ |
| `target` | int | 0 (Real) or 1 (Fake) | ✅ |
| `gender` | string | 'Male', 'Female' | ✅ |
| `skin_tone` | int | 1-10 (Monk Scale) | ✅ |

### Data Quality Observations:
✅ **Perfectly balanced distributions:**
- 50/50 Real vs Fake
- 50/50 Male vs Female  
- Equal samples across all 10 Monk Skin Tones
- Well-distributed across generation methods

✅ **No missing values** in any column  
✅ **Consistent naming conventions**  
✅ **Proper demographic stratification**

---

## Tools Created

The following verification tools have been created:

1. **`verify_dataset.py`** - Simple existence checker
2. **`verify_and_sample_dataset.py`** - Comprehensive verification with sampling
   - Run: `python verify_and_sample_dataset.py`
   - Outputs: Detailed report with sample paths for visual inspection

---

## Conclusion

**CSV Metadata:** ✅ Excellent quality, well-balanced, ready for training  
**Image Files:** ❌ Not present on disk - **Download required**

The annotation files are production-ready with perfect demographic balance and proper fairness considerations built into the dataset structure. Once the actual images are downloaded from the AI-Face FairnessBench dataset, the training pipeline can proceed immediately.

---

## Next Steps

1. Download AI-Face dataset images (35,000 images, ~3-5GB estimated)
2. Re-run `python verify_and_sample_dataset.py` to confirm extraction
3. Manually verify 10 samples per category using tool output
4. Proceed with training: `python train.py --annotations dataset/train_subset_mapped.csv`

---

**Report Generated:** September 1, 2026  
**Verification Tool:** `verify_and_sample_dataset.py`
