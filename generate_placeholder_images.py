"""
Generate placeholder images for testing the dataset pipeline
WITHOUT downloading the full AI-Face dataset (35,000 images).

WARNING: These are synthetic placeholders, NOT real deepfake detection data.
Use only for testing code, not for actual model training/evaluation.
"""
import os
from PIL import Image, ImageDraw, ImageFont
import random
import pandas as pd
from pathlib import Path

def create_placeholder_face_image(width=224, height=224, text="Sample", seed=42):
    """Create a simple placeholder image with text"""
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Create image with random background color
    bg_color = (
        random.randint(200, 255),
        random.randint(200, 255),
        random.randint(200, 255)
    )
    img = Image.new('RGB', (width, height), color=bg_color)
    
    draw = ImageDraw.Draw(img)
    
    # Draw a simple "face" shape
    face_color = (
        random.randint(180, 220),
        random.randint(140, 180),
        random.randint(120, 160)
    )
    
    # Face oval
    face_bbox = [width//4, height//4, 3*width//4, 3*height//4]
    draw.ellipse(face_bbox, fill=face_color, outline=(100, 100, 100), width=2)
    
    # Eyes
    eye_size = 10
    left_eye = [width//3 - eye_size, height//3 - eye_size, 
                width//3 + eye_size, height//3 + eye_size]
    right_eye = [2*width//3 - eye_size, height//3 - eye_size,
                 2*width//3 + eye_size, height//3 + eye_size]
    
    draw.ellipse(left_eye, fill=(50, 50, 50))
    draw.ellipse(right_eye, fill=(50, 50, 50))
    
    # Mouth
    mouth_bbox = [width//3, 2*height//3 - 10, 2*width//3, 2*height//3 + 10]
    draw.arc(mouth_bbox, start=0, end=180, fill=(50, 50, 50), width=2)
    
    # Add text label
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    # Text at bottom
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    text_x = (width - text_width) // 2
    text_y = height - text_height - 10
    
    draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)
    
    return img


def generate_images_from_csv(csv_path, num_samples=None, force_all=False):
    """
    Generate placeholder images for paths specified in CSV
    
    Args:
        csv_path: Path to CSV file
        num_samples: Number of images to generate (None = all)
        force_all: If True, generate all images; if False, ask for confirmation
    """
    print(f"\n{'='*80}")
    print(f"Generating placeholder images for: {csv_path}")
    print(f"{'='*80}\n")
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    if num_samples:
        df = df.head(num_samples)
    
    total = len(df)
    
    if not force_all and total > 100:
        print(f"⚠️  WARNING: This will generate {total:,} placeholder images.")
        print(f"   This is for TESTING ONLY. Not suitable for real training.")
        print(f"   Consider using num_samples parameter to generate fewer images.")
        response = input(f"\n   Continue? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("   Cancelled.")
            return
    
    print(f"Generating {total:,} placeholder images...\n")
    
    created = 0
    skipped = 0
    
    for idx, row in df.iterrows():
        img_path = row['image_path']
        
        # Create directory structure
        img_dir = os.path.dirname(img_path)
        os.makedirs(img_dir, exist_ok=True)
        
        # Skip if already exists
        if os.path.exists(img_path):
            skipped += 1
            continue
        
        # Extract info for label
        label = 'REAL' if row['target'] == 0 else 'FAKE'
        parts = img_path.split('/')
        category = next((p for p in parts if p in ['Real', 'GANs', 'DMs', 'deepfakes']), 'Unknown')
        
        # Create image with label
        text_label = f"{label}\n{category}"
        img = create_placeholder_face_image(text=text_label, seed=idx)
        
        # Save
        img.save(img_path, quality=85)
        created += 1
        
        if (idx + 1) % 500 == 0:
            print(f"   Progress: {idx+1:,}/{total:,} images...")
    
    print(f"\n✅ Complete!")
    print(f"   Created: {created:,}")
    print(f"   Skipped (already exist): {skipped:,}")
    print(f"   Total: {total:,}")


def generate_sample_subset(csv_path, output_dir="sample_images", samples_per_category=10):
    """
    Generate a small subset of images for quick testing
    """
    print(f"\n{'='*80}")
    print(f"Generating SAMPLE subset for quick testing")
    print(f"{'='*80}\n")
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    # Sample by category
    categories = {
        'Real': df[df['image_path'].str.contains('/Real/')],
        'GANs': df[df['image_path'].str.contains('/GANs/')],
        'DMs': df[df['image_path'].str.contains('/DMs/')],
        'deepfakes': df[df['image_path'].str.contains('/deepfakes/')]
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    total_created = 0
    
    for cat_name, cat_df in categories.items():
        if len(cat_df) == 0:
            continue
        
        print(f"📁 {cat_name}: sampling {min(samples_per_category, len(cat_df))} images...")
        
        sampled = cat_df.head(samples_per_category)
        
        for idx, row in sampled.iterrows():
            # Create simple filename
            filename = f"{cat_name}_{idx:04d}.jpg"
            img_path = os.path.join(output_dir, filename)
            
            label = 'REAL' if row['target'] == 0 else 'FAKE'
            text_label = f"{label}\n{cat_name}\nMST:{row['skin_tone']}"
            
            img = create_placeholder_face_image(text=text_label, seed=idx)
            img.save(img_path, quality=85)
            total_created += 1
    
    print(f"\n✅ Created {total_created} sample images in: {output_dir}/")
    print(f"   Use these for quick code testing!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate placeholder images for testing")
    parser.add_argument("--mode", type=str, choices=['sample', 'full', 'custom'], 
                        default='sample',
                        help="Generation mode: sample (40 images), full (all), custom (specify)")
    parser.add_argument("--train_csv", type=str, 
                        default="dataset/train_subset_mapped_fixed.csv",
                        help="Training CSV path")
    parser.add_argument("--test_csv", type=str,
                        default="dataset/test_subset_mapped_fixed.csv", 
                        help="Test CSV path")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of images to generate (for custom mode)")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print(" PLACEHOLDER IMAGE GENERATOR")
    print("="*80)
    print("\n⚠️  WARNING: These are PLACEHOLDER images for CODE TESTING only!")
    print("   DO NOT use for actual model training or evaluation.")
    print("   Download real AI-Face dataset for production use.\n")
    
    if args.mode == 'sample':
        print("Mode: SAMPLE - Generating ~40 images per dataset for quick testing\n")
        generate_sample_subset(args.train_csv, "sample_images/train", samples_per_category=10)
        generate_sample_subset(args.test_csv, "sample_images/test", samples_per_category=10)
        
    elif args.mode == 'full':
        print("Mode: FULL - Generating ALL 35,000 placeholder images\n")
        print("⚠️  This will take several minutes and create ~500MB of images.")
        response = input("Continue? (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            generate_images_from_csv(args.train_csv, force_all=True)
            generate_images_from_csv(args.test_csv, force_all=True)
        else:
            print("Cancelled.")
    
    elif args.mode == 'custom':
        if args.num_samples:
            print(f"Mode: CUSTOM - Generating {args.num_samples} images per dataset\n")
            generate_images_from_csv(args.train_csv, num_samples=args.num_samples)
            generate_images_from_csv(args.test_csv, num_samples=args.num_samples)
        else:
            print("❌ Error: --num_samples required for custom mode")
    
    print("\n" + "="*80)
    print(" Next Steps:")
    print("="*80)
    print("1. Run: python verify_fixed_csvs.py")
    print("2. Test your training code with these placeholder images")
    print("3. Download real AI-Face dataset for actual training")
    print("="*80 + "\n")
