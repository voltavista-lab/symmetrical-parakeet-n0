#!/bin/bash
# AWS EC2 Deployment Script for TFT Model
# This script automates the setup and training on AWS EC2

set -e

echo "=========================================="
echo "AWS EC2 TFT Deployment Script"
echo "=========================================="

# Configuration
INSTANCE_NAME="${INSTANCE_NAME:-tft-training}"
INSTANCE_TYPE="${INSTANCE_TYPE:-t3.xlarge}"
REGION="${AWS_REGION:-us-east-1}"
KEY_NAME="${KEY_NAME:-}"
SHEET="${SHEET:-southeast}"
EPOCHS="${EPOCHS:-100}"

# Check if running on EC2 or local
if [ -f /sys/hypervisor/uuid ] && [ `head -c 3 /sys/hypervisor/uuid` == ec2 ]; then
    ON_EC2=true
    echo "Running on EC2 instance"
else
    ON_EC2=false
    echo "Running locally"
fi

# Function to setup EC2 instance
setup_ec2() {
    echo ""
    echo "Setting up EC2 instance..."

    # Update system
    echo "Updating system packages..."
    sudo apt update && sudo apt upgrade -y

    # Install dependencies
    echo "Installing dependencies..."
    sudo apt install -y python3-pip python3-venv git htop tmux wget

    # Clone repository
    echo "Cloning repository..."
    if [ ! -d "symmetrical-parakeet-n0" ]; then
        git clone https://github.com/voltavista-lab/symmetrical-parakeet-n0.git
    fi

    cd symmetrical-parakeet-n0
    git checkout claude/temporal-fusion-transformers-2u7Wh
    git pull origin claude/temporal-fusion-transformers-2u7Wh

    # Create virtual environment
    echo "Creating Python virtual environment..."
    python3 -m venv venv
    source venv/bin/activate

    # Upgrade pip
    pip install --upgrade pip

    # Install PyTorch (CPU version for cost efficiency)
    echo "Installing PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

    # Install other requirements
    echo "Installing other requirements..."
    pip install pytorch-forecasting pytorch-lightning pandas numpy matplotlib scikit-learn openpyxl xlrd

    echo "Setup complete!"
}

# Function to run training
run_training() {
    echo ""
    echo "=========================================="
    echo "Starting TFT Training"
    echo "=========================================="
    echo "Sheet: $SHEET"
    echo "Epochs: $EPOCHS"
    echo "=========================================="
    echo ""

    cd symmetrical-parakeet-n0
    source venv/bin/activate

    # Run training
    python tft_train.py --sheet "$SHEET" --epochs "$EPOCHS"

    echo ""
    echo "Training complete!"
    echo "Results saved in current directory"
}

# Function to download results
download_results() {
    echo ""
    echo "Results files:"
    ls -lh tft_results_*.csv tft_*.png 2>/dev/null || echo "No results found"

    echo ""
    echo "To download results to your local machine, run:"
    echo "scp -i your-key.pem ubuntu@\$INSTANCE_IP:~/symmetrical-parakeet-n0/tft_*.* ."
}

# Main execution
main() {
    if [ "$ON_EC2" = true ]; then
        # Running on EC2 - perform setup and training
        setup_ec2
        run_training
        download_results

        echo ""
        echo "=========================================="
        echo "IMPORTANT: Remember to terminate the EC2"
        echo "instance when done to avoid charges!"
        echo "=========================================="
    else
        # Running locally - provide instructions
        cat << 'EOF'

This script should be run ON the EC2 instance after you've created it.

To deploy on AWS EC2:

1. Create EC2 instance:
   - Go to AWS Console -> EC2 -> Launch Instance
   - Choose Ubuntu 22.04 LTS
   - Instance type: t3.xlarge or larger
   - Create or select key pair
   - Allow SSH in security group
   - Launch instance

2. Connect to instance:
   ssh -i your-key.pem ubuntu@your-instance-ip

3. Copy this script to the instance:
   scp -i your-key.pem deploy_aws_ec2.sh ubuntu@your-instance-ip:~/

4. Run the script on the instance:
   chmod +x deploy_aws_ec2.sh
   ./deploy_aws_ec2.sh

Alternatively, use AWS CLI to automate instance creation:

# Set your configuration
export KEY_NAME="your-key-pair-name"
export INSTANCE_TYPE="t3.xlarge"

# Then run this script with 'launch' command:
./deploy_aws_ec2.sh launch

EOF
    fi
}

# Run main function
main
