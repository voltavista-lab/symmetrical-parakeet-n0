# Cloud Deployment Quick Start Guide

Choose your preferred cloud platform and follow the quick start instructions.

## 🚀 Fastest Options (FREE!)

### Google Colab (Recommended for Beginners)

**Time to start:** 2 minutes | **Cost:** FREE

1. Open in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/voltavista-lab/symmetrical-parakeet-n0/blob/claude/temporal-fusion-transformers-2u7Wh/colab_setup.ipynb)

2. Enable GPU: Runtime → Change runtime type → GPU → Save

3. Run all cells

4. Download results when complete

**Pros:** No setup, free GPU, works in browser
**Cons:** Session limits (12 hours), may disconnect

---

### AWS SageMaker Studio Lab

**Time to start:** 1-3 days (approval) + 5 minutes | **Cost:** FREE

1. Sign up: https://studiolab.sagemaker.aws/ (requires approval)

2. Once approved, login and start GPU runtime

3. Upload files: `tft_implementation.ipynb`, `requirements.txt`, data file

4. Open terminal and run:
   ```bash
   pip install -r requirements.txt
   ```

5. Open and run `tft_implementation.ipynb`

**Pros:** Free, includes GPU, no credit card needed
**Cons:** Approval wait time, limited hours per session

---

## 💳 Paid Options (Better Performance)

### AWS EC2 (Best Control)

**Time to start:** 10-15 minutes | **Cost:** ~$0.17-0.53/hour

**Quick Start:**
```bash
# 1. Launch Ubuntu 22.04 instance (t3.xlarge or g4dn.xlarge for GPU)
# 2. SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# 3. Run setup script
git clone https://github.com/voltavista-lab/symmetrical-parakeet-n0.git
cd symmetrical-parakeet-n0
git checkout claude/temporal-fusion-transformers-2u7Wh
chmod +x deploy_aws_ec2.sh
./deploy_aws_ec2.sh

# 4. Training starts automatically
# 5. Results saved in current directory
```

[Full AWS Guide →](DEPLOYMENT_AWS.md)

---

### GCP Compute Engine (Best for Google Users)

**Time to start:** 10-15 minutes | **Cost:** ~$0.19-0.53/hour

**Quick Start:**
```bash
# 1. Create Ubuntu VM (n1-standard-4 or with T4 GPU)
# From GCP Console or:
gcloud compute instances create tft-training \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud

# 2. SSH into instance
gcloud compute ssh tft-training --zone=us-central1-a

# 3. Run setup script
git clone https://github.com/voltavista-lab/symmetrical-parakeet-n0.git
cd symmetrical-parakeet-n0
git checkout claude/temporal-fusion-transformers-2u7Wh
chmod +x deploy_gcp_vm.sh
./deploy_gcp_vm.sh

# 4. Training starts automatically
```

[Full GCP Guide →](DEPLOYMENT_GCP.md)

---

## 🎯 Which Option Should I Choose?

### For Learning & Experimentation
→ **Google Colab** or **AWS SageMaker Studio Lab** (Both FREE)

### For Serious Development
→ **AWS EC2** with GPU (g4dn.xlarge) or **GCP Compute Engine** with T4 GPU

### For Production & Scale
→ **AWS SageMaker Training Jobs** or **GCP Vertex AI**

---

## 💰 Cost Comparison

| Platform | Instance Type | GPU | Cost/Hour | Training Time | Total Cost |
|----------|--------------|-----|-----------|---------------|------------|
| **Colab** | - | Free T4 | $0.00 | 30-60 min | **$0.00** |
| **SageMaker Lab** | - | Free | $0.00 | 30-60 min | **$0.00** |
| AWS EC2 | t3.xlarge | No | $0.17 | 2-4 hrs | $0.34-0.68 |
| AWS EC2 | g4dn.xlarge | T4 | $0.53 | 30-60 min | $0.26-0.53 |
| GCP VM | n1-standard-4 | No | $0.19 | 2-4 hrs | $0.38-0.76 |
| GCP VM | n1-std-4 + T4 | T4 | $0.53 | 30-60 min | $0.26-0.53 |

**Tip:** Use spot/preemptible instances for up to 80% savings!

---

## 📊 Training Time Estimates

| Dataset | CPU (no GPU) | GPU (T4) | GPU (V100) |
|---------|--------------|----------|------------|
| Southeast (1139 rows) | 2-4 hours | 30-60 min | 15-30 min |
| All submarkets | 8-16 hours | 2-4 hours | 1-2 hours |

---

## 🔥 Super Quick Start (Copy-Paste)

### Google Colab
```python
# In a new Colab notebook, paste and run:
!git clone https://github.com/voltavista-lab/symmetrical-parakeet-n0.git
%cd symmetrical-parakeet-n0
!git checkout claude/temporal-fusion-transformers-2u7Wh
!pip install -q pytorch-forecasting pytorch-lightning pandas openpyxl xlrd
!python tft_train.py --sheet southeast --epochs 100
```

### AWS EC2 (after SSH)
```bash
curl -O https://raw.githubusercontent.com/voltavista-lab/symmetrical-parakeet-n0/claude/temporal-fusion-transformers-2u7Wh/deploy_aws_ec2.sh
chmod +x deploy_aws_ec2.sh
./deploy_aws_ec2.sh
```

### GCP Compute Engine (after SSH)
```bash
curl -O https://raw.githubusercontent.com/voltavista-lab/symmetrical-parakeet-n0/claude/temporal-fusion-transformers-2u7Wh/deploy_gcp_vm.sh
chmod +x deploy_gcp_vm.sh
./deploy_gcp_vm.sh
```

---

## 🛑 IMPORTANT: Stop Charges!

**AWS:**
- EC2: Terminate instance from Console when done
- SageMaker Notebook: Stop instance when done
- SageMaker Training: Automatically stops after completion

**GCP:**
- Compute Engine: Delete instance when done
  ```bash
  gcloud compute instances delete INSTANCE_NAME --zone=ZONE
  ```
- Vertex AI Workbench: Stop instance when done

**Always check your billing console after use!**

---

## 📁 Files in This Repository

| File | Purpose |
|------|---------|
| `tft_implementation.ipynb` | Interactive Jupyter notebook |
| `tft_train.py` | Command-line training script |
| `colab_setup.ipynb` | Google Colab ready notebook |
| `deploy_aws_ec2.sh` | AWS EC2 automated setup |
| `deploy_gcp_vm.sh` | GCP VM automated setup |
| `DEPLOYMENT_AWS.md` | Complete AWS deployment guide |
| `DEPLOYMENT_GCP.md` | Complete GCP deployment guide |
| `requirements.txt` | Python dependencies |
| `README_TFT.md` | Main documentation |

---

## 🆘 Need Help?

1. **Check the full guides:**
   - [AWS Deployment Guide](DEPLOYMENT_AWS.md)
   - [GCP Deployment Guide](DEPLOYMENT_GCP.md)
   - [Main TFT Documentation](README_TFT.md)

2. **Common issues:**
   - Out of memory → Reduce `--batch-size 32` or `--hidden-size 32`
   - Slow training → Enable GPU or use smaller `--encoder-length`
   - Dependencies error → Try using conda instead of pip

3. **Open an issue:** https://github.com/voltavista-lab/symmetrical-parakeet-n0/issues

---

## 🎓 Learning Path

1. **Start:** Google Colab (free, no setup)
2. **Experiment:** AWS/GCP with CPU instance (cheap)
3. **Optimize:** AWS/GCP with GPU instance (faster)
4. **Scale:** Managed training jobs (production)

---

## ✅ Next Steps After Training

1. Download results files:
   - `tft_results_{submarket}.csv` - Predictions
   - `tft_predictions_{submarket}.png` - Visualization
   - `tft_model_{submarket}.pt` - Trained model

2. Compare with LSTM results (automatic if available)

3. Experiment with hyperparameters:
   - Encoder length: `--encoder-length 26` to `104`
   - Hidden size: `--hidden-size 32` to `256`
   - Epochs: `--epochs 50` to `200`

4. Train on other submarkets:
   ```bash
   python tft_train.py --sheet northeast
   python tft_train.py --sheet north
   python tft_train.py --sheet south
   ```

Happy training! 🚀
