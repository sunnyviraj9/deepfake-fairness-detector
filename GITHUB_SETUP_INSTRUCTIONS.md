# GitHub Repository Setup Instructions

## ✅ Git Repository Created Locally

Your project has been initialized as a Git repository with all files committed.

**Repository Details:**
- Branch: `main`
- Commit: Initial commit with 36 files
- Remote URL: `https://github.com/sunnyviraj9/deepfake-fairness-detector.git`

---

## 🚀 Step-by-Step: Push to GitHub

### Step 1: Create Repository on GitHub

1. Go to: https://github.com/new
2. Log in as: **@sunnyviraj9**
3. Repository name: **`deepfake-fairness-detector`**
4. Description: `Deepfake Detection & Algorithmic Fairness Benchmark with Monk Skin Tone Scale`
5. **Keep it Public** (or Private, your choice)
6. **DO NOT** initialize with README (we already have one)
7. **DO NOT** add .gitignore (we already have one)
8. Click **"Create repository"**

### Step 2: Push Your Code

After creating the repository on GitHub, run:

```bash
# Push to GitHub
git push -u origin main
```

**If prompted for credentials:**
- GitHub no longer accepts passwords for command-line operations
- You'll need a **Personal Access Token (PAT)**

---

## 🔑 Setting Up GitHub Authentication

### Option 1: Personal Access Token (Recommended)

1. **Generate Token:**
   - Go to: https://github.com/settings/tokens
   - Click "Generate new token" → "Generate new token (classic)"
   - Note: "Git operations for deepfake-detector"
   - Expiration: 90 days (or your preference)
   - Scopes: Check **`repo`** (full control of private repositories)
   - Click "Generate token"
   - **Copy the token** (you won't see it again!)

2. **Use Token When Pushing:**
   ```bash
   git push -u origin main
   ```
   - Username: `sunnyviraj9`
   - Password: `paste_your_token_here`

3. **Save Credentials (Optional):**
   ```bash
   # Store credentials so you don't need to enter them every time
   git config --global credential.helper store
   ```

### Option 2: GitHub CLI (Alternative)

```bash
# Install GitHub CLI: https://cli.github.com/
# Then authenticate:
gh auth login

# Push using:
git push -u origin main
```

### Option 3: SSH (Advanced)

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"

# Add to ssh-agent
ssh-add ~/.ssh/id_ed25519

# Copy public key
cat ~/.ssh/id_ed25519.pub

# Add to GitHub: https://github.com/settings/keys

# Change remote to SSH
git remote set-url origin git@github.com:sunnyviraj9/deepfake-fairness-detector.git

# Push
git push -u origin main
```

---

## 📋 What's Included in This Repository

### Core Files (36 files committed):
- ✅ Source code: `train.py`, `evaluate.py`, `inference.py`, `benchmark.py`
- ✅ Models: Spatial, Frequency, Dual-Stream detectors
- ✅ Utilities: Dataset loader, metrics, fairness loss
- ✅ Tests: Unit tests for models, dataset, metrics
- ✅ Documentation: README, dataset verification reports
- ✅ Dataset tools: CSV verification, image download helpers
- ✅ CSV annotations: Train/test subsets (fixed versions)
- ✅ Configuration: default_config.yaml

### Excluded from Git (.gitignore):
- ❌ Virtual environments (venv/, .venv/)
- ❌ Model checkpoints (*.pt, *.pth)
- ❌ Image directories (images/, AI-Face-downloaded/)
- ❌ Results and logs
- ❌ Python cache (__pycache__/, *.pyc)

---

## 🎯 After Pushing to GitHub

### Add Repository Details:

**On GitHub repository page:**
1. Add **Topics/Tags**: `deepfake-detection`, `fairness-benchmark`, `monk-skin-tone`, `pytorch`, `computer-vision`, `cvpr-2025`
2. Update **About section**: Add description and website
3. Add **License**: Consider MIT or Apache 2.0

### Create Additional Files (Optional):

```bash
# Create a more detailed contributing guide
# Add issue templates
# Set up GitHub Actions for CI/CD
```

### Share Your Repository:

Your public URL will be:
```
https://github.com/sunnyviraj9/deepfake-fairness-detector
```

---

## 📝 Quick Reference Commands

```bash
# Check status
git status

# Stage changes
git add .

# Commit changes
git commit -m "Your commit message"

# Push to GitHub
git push origin main

# Pull latest changes
git pull origin main

# Create new branch
git checkout -b feature-name

# View commit history
git log --oneline

# View remote info
git remote -v
```

---

## 🔄 Future Workflow

After initial push, use this workflow:

```bash
# 1. Make changes to your code
# 2. Stage changes
git add .

# 3. Commit with descriptive message
git commit -m "Add frequency domain improvements"

# 4. Push to GitHub
git push origin main
```

---

## 🆘 Troubleshooting

### Error: "Repository not found"
- Make sure you created the repository on GitHub first
- Check repository name matches: `deepfake-fairness-detector`

### Error: "Authentication failed"
- Use Personal Access Token instead of password
- Make sure token has `repo` scope

### Error: "Updates were rejected"
- Someone else pushed first, or you edited on GitHub
- Pull first: `git pull origin main --rebase`
- Then push: `git push origin main`

### Large File Warning
- Git warns about files >50MB
- Model checkpoints should be in .gitignore
- Use Git LFS for large files if needed

---

## 📧 Next Steps

1. **Create repo on GitHub** (Step 1 above)
2. **Push your code**: `git push -u origin main`
3. **Verify on GitHub**: Visit your repository URL
4. **Add collaborators** (if needed): Settings → Collaborators
5. **Set up branch protection** (optional): Settings → Branches

---

## 🎉 Repository Features to Add

- [ ] GitHub Actions for automated testing
- [ ] Code quality badges (coverage, build status)
- [ ] Issue templates for bugs and features
- [ ] Pull request template
- [ ] CONTRIBUTING.md guide
- [ ] CODE_OF_CONDUCT.md
- [ ] LICENSE file

---

**Your repository is ready to push!** 🚀

Run: `git push -u origin main`
