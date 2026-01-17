# AWS Deployment Guide for TFT Model

This guide covers multiple ways to deploy and run the TFT model on AWS, from simple to advanced.

## Table of Contents
1. [Quick Start: AWS SageMaker Studio Lab (Free)](#option-1-aws-sagemaker-studio-lab-free)
2. [AWS EC2 Instance](#option-2-aws-ec2-instance)
3. [AWS SageMaker Training Jobs](#option-3-aws-sagemaker-training-jobs)
4. [AWS SageMaker Notebooks](#option-4-aws-sagemaker-notebooks)
5. [Cost Optimization](#cost-optimization)
6. [Troubleshooting](#troubleshooting)

---

## Option 1: AWS SageMaker Studio Lab (FREE)

**Best for:** Learning and experimentation
**Cost:** FREE (no credit card required)
**GPU:** Free GPU access (limited hours)

### Steps:

1. **Sign up for SageMaker Studio Lab**
   - Go to: https://studiolab.sagemaker.aws/
   - Request free account (approval takes 1-3 days)

2. **Launch Studio Lab**
   - Login and start your runtime
   - Choose GPU runtime for faster training

3. **Upload files**
   - Click upload button
   - Upload: `tft_implementation.ipynb`, `requirements.txt`, `FINAL_INPUTS_v2.xls`

4. **Install dependencies**
   ```bash
   # In a notebook cell or terminal
   !pip install -r requirements.txt
   ```

5. **Run the notebook**
   - Open `tft_implementation.ipynb`
   - Run all cells

**Pros:** Free, no setup, GPU included
**Cons:** Limited compute hours (4-12 hrs per session), not for production

---

## Option 2: AWS EC2 Instance

**Best for:** Full control, production workloads
**Cost:** ~$0.10-$3.00/hour (depending on instance type)

### 2.1 Quick Setup (CPU Instance - Cheapest)

```bash
# 1. Launch EC2 instance from AWS Console
#    - AMI: Ubuntu 22.04 LTS
#    - Instance type: t3.xlarge (4 vCPU, 16GB RAM) - ~$0.17/hour
#    - Storage: 30GB gp3
#    - Security group: Allow SSH (port 22)

# 2. Connect to instance
ssh -i your-key.pem ubuntu@your-instance-ip

# 3. Install dependencies
sudo apt update
sudo apt install -y python3-pip python3-venv git

# 4. Clone your repository
git clone https://github.com/voltavista-lab/symmetrical-parakeet-n0.git
cd symmetrical-parakeet-n0

# 5. Switch to TFT branch
git checkout claude/temporal-fusion-transformers-2u7Wh

# 6. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 7. Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 8. Install other requirements
pip install pytorch-forecasting pytorch-lightning pandas numpy matplotlib scikit-learn openpyxl xlrd

# 9. Run training
python tft_train.py --sheet southeast --epochs 100
```

### 2.2 GPU Setup (Faster Training)

```bash
# 1. Launch EC2 instance
#    - AMI: Deep Learning AMI (Ubuntu 20.04)
#    - Instance type: g4dn.xlarge (1 GPU, 4 vCPU, 16GB RAM) - ~$0.53/hour
#    - Storage: 50GB gp3
#    - Security group: Allow SSH

# 2. Connect to instance
ssh -i your-key.pem ubuntu@your-instance-ip

# 3. Activate conda environment (comes pre-installed in Deep Learning AMI)
source activate pytorch

# 4. Clone repository
git clone https://github.com/voltavista-lab/symmetrical-parakeet-n0.git
cd symmetrical-parakeet-n0
git checkout claude/temporal-fusion-transformers-2u7Wh

# 5. Install requirements
pip install pytorch-forecasting pandas openpyxl xlrd

# 6. Verify GPU
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"

# 7. Run training
python tft_train.py --sheet southeast --epochs 100
```

### 2.3 Automated Setup Script

Save as `setup_ec2.sh`:

```bash
#!/bin/bash
set -e

echo "=== TFT EC2 Setup Script ==="

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
chmod +x setup_ec2.sh
./setup_ec2.sh
```

### 2.4 Running Training in Background

Use `tmux` or `screen` to keep training running after disconnecting:

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

## Option 3: AWS SageMaker Training Jobs

**Best for:** Scalable, managed training
**Cost:** Pay only for training time (~$0.27-$5/hour)

### 3.1 Prepare Training Script

Create `train_sagemaker.py`:

```python
import os
import argparse
import sys

# SageMaker passes hyperparameters as arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument('--sheet', type=str, default='southeast')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--hidden-size', type=int, default=64)

    # SageMaker specific
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))

    args = parser.parse_args()

    # Import after parsing args
    from tft_train import main as run_training

    # Run training
    run_training()
```

### 3.2 SageMaker Training Script

Create `run_sagemaker_training.py`:

```python
import sagemaker
from sagemaker.pytorch import PyTorch

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()  # Or specify your IAM role ARN
bucket = sagemaker_session.default_bucket()

# Upload data to S3
data_location = sagemaker_session.upload_data(
    path='FINAL_INPUTS_v2.xls',
    bucket=bucket,
    key_prefix='tft-training/data'
)

# Configure PyTorch estimator
estimator = PyTorch(
    entry_point='tft_train.py',
    role=role,
    instance_type='ml.g4dn.xlarge',  # GPU instance
    instance_count=1,
    framework_version='2.0.0',
    py_version='py310',
    hyperparameters={
        'sheet': 'southeast',
        'epochs': 100,
        'hidden-size': 64
    },
    source_dir='.',
    dependencies=['requirements.txt']
)

# Start training
estimator.fit({'train': data_location})

print(f"Training job completed!")
print(f"Model artifacts: {estimator.model_data}")
```

Run with:
```bash
pip install sagemaker
python run_sagemaker_training.py
```

---

## Option 4: AWS SageMaker Notebooks

**Best for:** Interactive development with managed infrastructure
**Cost:** ~$0.05-$1/hour (pay per use)

### Steps:

1. **Create Notebook Instance**
   - Go to: AWS Console → SageMaker → Notebook instances
   - Click "Create notebook instance"
   - Name: `tft-training`
   - Instance type: `ml.t3.xlarge` (for CPU) or `ml.g4dn.xlarge` (for GPU)
   - IAM role: Create new or use existing
   - Click "Create"

2. **Open JupyterLab**
   - Wait for instance to be "InService"
   - Click "Open JupyterLab"

3. **Setup Environment**
   ```bash
   # Open terminal in JupyterLab
   cd SageMaker
   git clone https://github.com/voltavista-lab/symmetrical-parakeet-n0.git
   cd symmetrical-parakeet-n0
   git checkout claude/temporal-fusion-transformers-2u7Wh

   # Install requirements
   pip install -r requirements.txt
   ```

4. **Upload Data**
   - Upload `FINAL_INPUTS_v2.xls` via JupyterLab interface

5. **Run Notebook**
   - Open `tft_implementation.ipynb`
   - Run all cells

**Important:** Stop the notebook instance when not in use to avoid charges!

---

## Cost Optimization

### 1. Use Spot Instances (up to 90% savings)

For EC2:
```bash
# Launch spot instance instead of on-demand
# In AWS Console, select "Request Spot Instances" when launching
```

For SageMaker:
```python
estimator = PyTorch(
    # ... other parameters ...
    use_spot_instances=True,
    max_run=3600,  # Max training time
    max_wait=7200  # Max wait for spot capacity
)
```

### 2. Right-Size Your Instance

| Instance Type | vCPU | RAM | GPU | Cost/hr | Best For |
|---------------|------|-----|-----|---------|----------|
| t3.xlarge | 4 | 16GB | - | $0.17 | Small experiments |
| m5.xlarge | 4 | 16GB | - | $0.19 | CPU training |
| g4dn.xlarge | 4 | 16GB | 1x T4 | $0.53 | GPU training |
| p3.2xlarge | 8 | 61GB | 1x V100 | $3.06 | Fast GPU training |

### 3. Auto-Shutdown

For SageMaker Notebooks, create lifecycle config:
```bash
#!/bin/bash
# Auto-stop after 1 hour of idle time
echo "Setting up auto-shutdown..."
cat > /home/ec2-user/SageMaker/auto-shutdown.sh << 'EOF'
#!/bin/bash
IDLE_TIME=3600  # 1 hour
if [ $(who | wc -l) -eq 0 ]; then
    sudo shutdown -h now
fi
EOF
chmod +x /home/ec2-user/SageMaker/auto-shutdown.sh
(crontab -l 2>/dev/null; echo "*/5 * * * * /home/ec2-user/SageMaker/auto-shutdown.sh") | crontab -
```

### 4. Use S3 for Data Storage

Store data in S3 instead of expensive EBS:
```python
import pandas as pd
import boto3

# Read directly from S3
df = pd.read_excel('s3://your-bucket/FINAL_INPUTS_v2.xls')
```

### 5. Training Time Estimates

| Configuration | Approx. Training Time | Estimated Cost |
|---------------|----------------------|----------------|
| CPU (t3.xlarge) | 2-4 hours | $0.34-$0.68 |
| GPU (g4dn.xlarge) | 30-60 min | $0.26-$0.53 |
| GPU (p3.2xlarge) | 15-30 min | $0.77-$1.53 |

---

## Complete Example: EC2 with GPU

```bash
# 1. Launch g4dn.xlarge with Deep Learning AMI

# 2. Connect and setup
ssh -i key.pem ubuntu@instance-ip

# 3. Quick setup
source activate pytorch
git clone https://github.com/voltavista-lab/symmetrical-parakeet-n0.git
cd symmetrical-parakeet-n0
git checkout claude/temporal-fusion-transformers-2u7Wh
pip install pytorch-forecasting pandas openpyxl xlrd

# 4. Run training in background
tmux new -s training
python tft_train.py --sheet southeast --epochs 100

# 5. Detach (Ctrl+B, D) and disconnect
# Training continues in background

# 6. Reconnect later to check results
ssh -i key.pem ubuntu@instance-ip
tmux attach -t training

# 7. Download results
# On your local machine:
scp -i key.pem ubuntu@instance-ip:~/symmetrical-parakeet-n0/tft_results_*.csv .
scp -i key.pem ubuntu@instance-ip:~/symmetrical-parakeet-n0/tft_*.png .

# 8. IMPORTANT: Terminate instance when done!
# Go to EC2 Console → Instances → Select instance → Instance State → Terminate
```

---

## Monitoring Training

### Using TensorBoard

```bash
# Start TensorBoard in background
tmux new -s tensorboard
tensorboard --logdir lightning_logs --host 0.0.0.0 --port 6006

# Detach: Ctrl+B, D

# In EC2 security group, allow inbound port 6006
# Access at: http://your-instance-ip:6006
```

### Using CloudWatch (SageMaker)

Training metrics automatically logged to CloudWatch when using SageMaker Training Jobs.

---

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python tft_train.py --batch-size 32

# Or use smaller model
python tft_train.py --hidden-size 32
```

### CUDA Out of Memory
```bash
# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Or reduce encoder length
python tft_train.py --encoder-length 26
```

### Slow Training
```bash
# Verify GPU is being used
python -c "import torch; print(torch.cuda.is_available())"

# Use smaller dataset for testing
python tft_train.py --epochs 10 --encoder-length 12
```

### Dependencies Issues
```bash
# Use conda instead of pip
conda create -n tft python=3.10
conda activate tft
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install pytorch-forecasting pandas openpyxl xlrd
```

---

## Security Best Practices

1. **Use IAM Roles** instead of access keys
2. **Restrict Security Groups** to your IP only
3. **Use VPC** for production deployments
4. **Encrypt data** in S3 and EBS
5. **Enable CloudTrail** for audit logging

---

## Next Steps

1. Start with SageMaker Studio Lab (free) for testing
2. Move to EC2 for production
3. Use SageMaker Training Jobs for scale
4. Set up CI/CD pipeline with AWS CodePipeline

For questions, refer to:
- AWS SageMaker: https://docs.aws.amazon.com/sagemaker/
- AWS EC2: https://docs.aws.amazon.com/ec2/
