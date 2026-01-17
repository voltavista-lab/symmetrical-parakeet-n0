# GCP Deployment Guide for TFT Model

This guide covers multiple ways to deploy and run the TFT model on Google Cloud Platform (GCP).

## Table of Contents
1. [Quick Start: Google Colab (Free)](#option-1-google-colab-free)
2. [GCP Compute Engine VM](#option-2-gcp-compute-engine-vm)
3. [GCP Vertex AI Training Jobs](#option-3-vertex-ai-training-jobs)
4. [GCP Vertex AI Workbench](#option-4-vertex-ai-workbench)
5. [Cost Optimization](#cost-optimization)
6. [Troubleshooting](#troubleshooting)

---

## Option 1: Google Colab (FREE)

**Best for:** Quick experiments and learning
**Cost:** FREE (Colab) or $9.99/month (Colab Pro)
**GPU:** Free GPU access (limited)

### Steps:

1. **Open Google Colab**
   - Go to: https://colab.research.google.com/

2. **Mount Google Drive**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **Upload files to Google Drive**
   - Upload: `tft_implementation.ipynb`, `FINAL_INPUTS_v2.xls`
   - Or clone from GitHub:
   ```python
   !git clone https://github.com/voltavista-lab/symmetrical-parakeet-n0.git
   %cd symmetrical-parakeet-n0
   !git checkout claude/temporal-fusion-transformers-2u7Wh
   ```

4. **Enable GPU**
   - Runtime → Change runtime type → GPU → Save

5. **Install dependencies**
   ```python
   !pip install pytorch-forecasting pytorch-lightning pandas openpyxl xlrd
   ```

6. **Run training**
   ```python
   !python tft_train.py --sheet southeast --epochs 100
   ```

7. **Download results**
   ```python
   from google.colab import files
   files.download('tft_results_southeast.csv')
   files.download('tft_predictions_southeast.png')
   ```

**Pros:** Free, no setup, GPU included, easy to share
**Cons:** Limited session time (12 hours max), can disconnect, not for production

---

## Option 2: GCP Compute Engine VM

**Best for:** Full control, production workloads
**Cost:** ~$0.10-$2.50/hour (depending on instance type)

### 2.1 Quick Setup (CPU Instance - Cheapest)

```bash
# 1. Create VM from Cloud Console or gcloud CLI

# Using gcloud CLI:
gcloud compute instances create tft-training-vm \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB \
  --boot-disk-type=pd-standard

# 2. SSH into instance
gcloud compute ssh tft-training-vm --zone=us-central1-a

# 3. Install dependencies
sudo apt update
sudo apt install -y python3-pip python3-venv git

# 4. Clone repository
git clone https://github.com/voltavista-lab/symmetrical-parakeet-n0.git
cd symmetrical-parakeet-n0
git checkout claude/temporal-fusion-transformers-2u7Wh

# 5. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 6. Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 7. Install other requirements
pip install pytorch-forecasting pytorch-lightning pandas numpy matplotlib scikit-learn openpyxl xlrd

# 8. Run training
python tft_train.py --sheet southeast --epochs 100
```

### 2.2 GPU Setup (Faster Training)

```bash
# 1. Create GPU instance
gcloud compute instances create tft-training-gpu \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=50GB \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --metadata="install-nvidia-driver=True"

# 2. SSH into instance
gcloud compute ssh tft-training-gpu --zone=us-central1-a

# 3. Verify GPU
nvidia-smi

# 4. Clone repository
git clone https://github.com/voltavista-lab/symmetrical-parakeet-n0.git
cd symmetrical-parakeet-n0
git checkout claude/temporal-fusion-transformers-2u7Wh

# 5. Install requirements (PyTorch already installed in image)
pip install pytorch-forecasting pandas openpyxl xlrd

# 6. Verify GPU in PyTorch
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"

# 7. Run training
python tft_train.py --sheet southeast --epochs 100
```

### 2.3 Automated Setup Script

Save as `setup_gcp.sh`:

```bash
#!/bin/bash
set -e

echo "=== TFT GCP Setup Script ==="

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and tools
sudo apt install -y python3-pip python3-venv git htop tmux

# Clone repository
if [ ! -d "symmetrical-parakeet-n0" ]; then
    git clone https://github.com/voltavista-lab/symmetrical-parakeet-n0.git
fi

cd symmetrical-parakeet-n0
git checkout claude/temporal-fusion-transformers-2u7Wh

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install requirements
pip install -r requirements.txt

echo "=== Setup Complete! ==="
echo "To activate environment: source venv/bin/activate"
echo "To run training: python tft_train.py --sheet southeast"
```

Run with:
```bash
chmod +x setup_gcp.sh
./setup_gcp.sh
```

### 2.4 Running Training in Background

```bash
# Start tmux session
tmux new -s tft_training

# Run training
source venv/bin/activate
python tft_train.py --sheet southeast --epochs 100

# Detach: Press Ctrl+B, then D
# Reattach later: tmux attach -t tft_training
```

---

## Option 3: Vertex AI Training Jobs

**Best for:** Scalable, managed training
**Cost:** Pay only for training time (~$0.35-$4/hour)

### 3.1 Prepare Training Script

Create `train_vertex.py`:

```python
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument('--sheet', type=str, default='southeast')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--hidden-size', type=int, default=64)

    # Vertex AI specific
    parser.add_argument('--model-dir', type=str, default=os.environ.get('AIP_MODEL_DIR'))

    args = parser.parse_args()

    # Import and run training
    from tft_train import main as run_training
    run_training()
```

### 3.2 Create Dockerfile

Create `Dockerfile`:

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy training code
COPY tft_train.py .
COPY FINAL_INPUTS_v2.xls .

# Set entrypoint
ENTRYPOINT ["python", "tft_train.py"]
```

### 3.3 Build and Push Container

```bash
# Set variables
PROJECT_ID="your-gcp-project-id"
REGION="us-central1"
IMAGE_NAME="tft-training"
IMAGE_URI="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:latest"

# Build container
docker build -t ${IMAGE_URI} .

# Push to Google Container Registry
docker push ${IMAGE_URI}

# Or use Cloud Build
gcloud builds submit --tag ${IMAGE_URI}
```

### 3.4 Submit Training Job

```bash
# Using gcloud CLI
gcloud ai custom-jobs create \
  --region=${REGION} \
  --display-name=tft-training-job \
  --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,container-image-uri=${IMAGE_URI} \
  --args="--sheet=southeast,--epochs=100"

# With GPU
gcloud ai custom-jobs create \
  --region=${REGION} \
  --display-name=tft-training-job-gpu \
  --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,container-image-uri=${IMAGE_URI} \
  --args="--sheet=southeast,--epochs=100"
```

### 3.5 Python SDK Approach

Create `run_vertex_training.py`:

```python
from google.cloud import aiplatform

# Initialize Vertex AI
aiplatform.init(project='your-project-id', location='us-central1')

# Create custom training job
job = aiplatform.CustomContainerTrainingJob(
    display_name='tft-training',
    container_uri='gcr.io/your-project-id/tft-training:latest',
)

# Run training job
job.run(
    machine_type='n1-standard-4',
    accelerator_type='NVIDIA_TESLA_T4',
    accelerator_count=1,
    args=['--sheet', 'southeast', '--epochs', '100'],
    replica_count=1,
)

print(f"Training job completed!")
```

Run with:
```bash
pip install google-cloud-aiplatform
python run_vertex_training.py
```

---

## Option 4: Vertex AI Workbench

**Best for:** Interactive development with managed infrastructure
**Cost:** ~$0.15-$1.50/hour

### Steps:

1. **Create Workbench Instance**
   ```bash
   gcloud workbench instances create tft-notebook \
     --location=us-central1-a \
     --machine-type=n1-standard-4 \
     --accelerator-type=NVIDIA_TESLA_T4 \
     --accelerator-core-count=1
   ```

   Or via Console:
   - Go to: Vertex AI → Workbench → User-Managed Notebooks
   - Click "New Notebook"
   - Select PyTorch environment
   - Choose machine type (e.g., n1-standard-4 with T4 GPU)

2. **Open JupyterLab**
   - Click "Open JupyterLab" when instance is ready

3. **Setup in Terminal**
   ```bash
   cd /home/jupyter
   git clone https://github.com/voltavista-lab/symmetrical-parakeet-n0.git
   cd symmetrical-parakeet-n0
   git checkout claude/temporal-fusion-transformers-2u7Wh

   # Install requirements
   pip install -r requirements.txt
   ```

4. **Upload Data**
   - Use JupyterLab file upload for `FINAL_INPUTS_v2.xls`

5. **Run Notebook**
   - Open `tft_implementation.ipynb`
   - Run all cells

**Important:** Stop the instance when not in use!

---

## Cost Optimization

### 1. Use Preemptible VMs (up to 80% savings)

```bash
# Create preemptible instance
gcloud compute instances create tft-training-preemptible \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --preemptible \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release
```

**Important:** Preemptible VMs can be terminated at any time. Save checkpoints frequently!

### 2. Right-Size Your Instance

| Machine Type | vCPU | RAM | GPU | Cost/hr | Best For |
|--------------|------|-----|-----|---------|----------|
| n1-standard-2 | 2 | 7.5GB | - | $0.095 | Small experiments |
| n1-standard-4 | 4 | 15GB | - | $0.19 | CPU training |
| n1-standard-4 + T4 | 4 | 15GB | 1x T4 | $0.53 | GPU training |
| n1-standard-8 + V100 | 8 | 30GB | 1x V100 | $2.48 | Fast GPU training |

### 3. Use Committed Use Discounts

For long-term usage, purchase 1-year or 3-year commitments for up to 57% discount.

### 4. Auto-Shutdown Script

Create `/etc/cron.hourly/auto-shutdown`:

```bash
#!/bin/bash
IDLE_TIME=3600  # 1 hour in seconds
CURRENT_IDLE=$(who -s | wc -l)

if [ $CURRENT_IDLE -eq 0 ]; then
    shutdown -h now
fi
```

Or use this in Vertex AI Workbench:
```bash
# Set idle shutdown (in minutes)
gcloud workbench instances update tft-notebook \
  --location=us-central1-a \
  --idle-shutdown-timeout=60
```

### 5. Use Cloud Storage for Data

```bash
# Upload data to Cloud Storage
gsutil cp FINAL_INPUTS_v2.xls gs://your-bucket/data/

# In your code, read directly from GCS
import pandas as pd
df = pd.read_excel('gs://your-bucket/data/FINAL_INPUTS_v2.xls')
```

### 6. Training Time Estimates

| Configuration | Approx. Training Time | Estimated Cost |
|---------------|----------------------|----------------|
| CPU (n1-standard-4) | 2-4 hours | $0.38-$0.76 |
| GPU (T4) | 30-60 min | $0.26-$0.53 |
| GPU (V100) | 15-30 min | $0.62-$1.24 |

---

## Complete Example: Compute Engine with GPU

```bash
# 1. Create GPU instance
gcloud compute instances create tft-gpu \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=50GB \
  --maintenance-policy=TERMINATE \
  --metadata="install-nvidia-driver=True"

# 2. SSH into instance
gcloud compute ssh tft-gpu --zone=us-central1-a

# 3. Setup
git clone https://github.com/voltavista-lab/symmetrical-parakeet-n0.git
cd symmetrical-parakeet-n0
git checkout claude/temporal-fusion-transformers-2u7Wh
pip install pytorch-forecasting pandas openpyxl xlrd

# 4. Run training in background
tmux new -s training
python tft_train.py --sheet southeast --epochs 100

# 5. Detach (Ctrl+B, D) and disconnect

# 6. Reconnect later
gcloud compute ssh tft-gpu --zone=us-central1-a
tmux attach -t training

# 7. Download results (from local machine)
gcloud compute scp tft-gpu:~/symmetrical-parakeet-n0/tft_results_*.csv . --zone=us-central1-a
gcloud compute scp tft-gpu:~/symmetrical-parakeet-n0/tft_*.png . --zone=us-central1-a

# 8. IMPORTANT: Delete instance when done!
gcloud compute instances delete tft-gpu --zone=us-central1-a
```

---

## Monitoring Training

### Using TensorBoard

```bash
# Start TensorBoard
tmux new -s tensorboard
tensorboard --logdir lightning_logs --host 0.0.0.0 --port 6006

# Create firewall rule (one time)
gcloud compute firewall-rules create tensorboard \
  --allow tcp:6006 \
  --source-ranges 0.0.0.0/0 \
  --description "TensorBoard access"

# Get external IP
gcloud compute instances describe tft-gpu --zone=us-central1-a \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)'

# Access at: http://EXTERNAL_IP:6006
```

### Using Cloud Logging (Vertex AI)

Training logs automatically sent to Cloud Logging when using Vertex AI Training Jobs.

---

## GCS Integration

### Upload Data to Cloud Storage

```bash
# Create bucket
gsutil mb gs://your-tft-bucket

# Upload data
gsutil cp FINAL_INPUTS_v2.xls gs://your-tft-bucket/data/

# List files
gsutil ls gs://your-tft-bucket/data/
```

### Modify Code to Use GCS

```python
# In tft_train.py, modify data loading:
import gcsfs

# Use GCS path
CSV_PATH = 'gs://your-tft-bucket/data/FINAL_INPUTS_v2.xls'

# Or download first
!gsutil cp gs://your-tft-bucket/data/FINAL_INPUTS_v2.xls .
```

---

## Troubleshooting

### GPU Not Detected
```bash
# Check GPU quota
gcloud compute project-info describe --project=your-project-id

# Request GPU quota increase if needed
# Console → IAM & Admin → Quotas → Search "GPUs" → Request increase

# Verify driver installation
nvidia-smi

# Reinstall driver if needed
sudo /opt/deeplearning/install-driver.sh
```

### Out of Memory
```bash
# Reduce batch size
python tft_train.py --batch-size 32

# Or use smaller model
python tft_train.py --hidden-size 32
```

### Preemptible VM Terminated
```bash
# Training was interrupted, resume from checkpoint
# The TFT model automatically saves checkpoints
# Just restart training with same command

# For better handling, modify tft_train.py to resume from checkpoint
```

### Dependencies Issues
```bash
# Use conda environment
conda create -n tft python=3.10
conda activate tft
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install pytorch-forecasting pandas openpyxl xlrd
```

---

## Security Best Practices

1. **Use Service Accounts** instead of user credentials
2. **Restrict Firewall Rules** to your IP only
3. **Use VPC** for production deployments
4. **Enable Private Google Access** for VMs without external IPs
5. **Encrypt data** in Cloud Storage
6. **Enable Cloud Audit Logs**

---

## Comparison: Colab vs Compute Engine vs Vertex AI

| Feature | Colab | Compute Engine | Vertex AI |
|---------|-------|----------------|-----------|
| Cost | Free/$10/mo | $0.10-2.50/hr | $0.35-4/hr |
| GPU | Limited free | Pay per use | Pay per use |
| Setup | None | Manual | Minimal |
| Control | Limited | Full | Managed |
| Production | No | Yes | Yes |
| Auto-scaling | No | No | Yes |
| Best For | Learning | Development | Production |

---

## Next Steps

1. **Start with Google Colab** for free testing
2. **Move to Compute Engine** for development
3. **Use Vertex AI Training Jobs** for production at scale
4. **Set up CI/CD** with Cloud Build

For questions, refer to:
- Vertex AI: https://cloud.google.com/vertex-ai/docs
- Compute Engine: https://cloud.google.com/compute/docs
- Google Colab: https://colab.research.google.com/
