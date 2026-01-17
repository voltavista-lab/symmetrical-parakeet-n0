#!/bin/bash
# GCP Compute Engine Deployment Script for TFT Model
# This script automates the setup and training on GCP VM

set -e

echo "=========================================="
echo "GCP Compute Engine TFT Deployment Script"
echo "=========================================="

# Configuration
INSTANCE_NAME="${INSTANCE_NAME:-tft-training-vm}"
MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-4}"
ZONE="${ZONE:-us-central1-a}"
PROJECT_ID="${PROJECT_ID:-}"
SHEET="${SHEET:-southeast}"
EPOCHS="${EPOCHS:-100}"

# Detect if running on GCP
if curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/ &> /dev/null; then
    ON_GCP=true
    echo "Running on GCP instance"
else
    ON_GCP=false
    echo "Running locally"
fi

# Function to create GCP instance
create_instance() {
    echo ""
    echo "Creating GCP Compute Engine instance..."

    if [ -z "$PROJECT_ID" ]; then
        echo "Error: PROJECT_ID environment variable not set"
        echo "Please set it with: export PROJECT_ID=your-project-id"
        exit 1
    fi

    gcloud compute instances create "$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --image-family=ubuntu-2204-lts \
        --image-project=ubuntu-os-cloud \
        --boot-disk-size=50GB \
        --boot-disk-type=pd-standard \
        --metadata-from-file=startup-script=<(cat <<'STARTUP_SCRIPT'
#!/bin/bash
# This runs automatically on instance creation
apt update
apt install -y python3-pip python3-venv git
STARTUP_SCRIPT
)

    echo ""
    echo "Instance created successfully!"
    echo "Connect with: gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID"
    echo ""
    echo "After connecting, run this script again to set up training:"
    echo "./deploy_gcp_vm.sh"
}

# Function to setup VM
setup_vm() {
    echo ""
    echo "Setting up GCP VM..."

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

    # Install PyTorch (CPU version)
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

    if [ "$ON_GCP" = true ]; then
        echo ""
        echo "To download results to your local machine, run:"
        echo "gcloud compute scp $INSTANCE_NAME:~/symmetrical-parakeet-n0/tft_*.* . --zone=$ZONE"
    fi
}

# Function to cleanup
cleanup() {
    echo ""
    echo "=========================================="
    echo "Cleanup Instructions"
    echo "=========================================="
    echo ""
    echo "To delete the instance and avoid charges:"
    echo "gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID"
    echo ""
    echo "IMPORTANT: Remember to delete the instance"
    echo "when done to avoid charges!"
    echo "=========================================="
}

# Main execution
main() {
    # Check command
    COMMAND="${1:-}"

    case "$COMMAND" in
        create)
            create_instance
            ;;
        *)
            if [ "$ON_GCP" = true ]; then
                # Running on GCP - perform setup and training
                setup_vm
                run_training
                download_results
                cleanup
            else
                # Running locally - provide instructions
                cat << 'EOF'

This script can be used in two ways:

METHOD 1: Create instance from local machine
-------------------------------------------
1. Install gcloud CLI: https://cloud.google.com/sdk/docs/install

2. Authenticate:
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID

3. Create instance with this script:
   export PROJECT_ID="your-project-id"
   export ZONE="us-central1-a"
   export MACHINE_TYPE="n1-standard-4"
   ./deploy_gcp_vm.sh create

4. SSH into instance:
   gcloud compute ssh tft-training-vm --zone=us-central1-a

5. Run setup and training:
   git clone https://github.com/voltavista-lab/symmetrical-parakeet-n0.git
   cd symmetrical-parakeet-n0
   git checkout claude/temporal-fusion-transformers-2u7Wh
   chmod +x deploy_gcp_vm.sh
   ./deploy_gcp_vm.sh

METHOD 2: Manual instance creation
-----------------------------------
1. Go to GCP Console -> Compute Engine -> Create Instance

2. Configure:
   - Name: tft-training-vm
   - Region: us-central1
   - Machine type: n1-standard-4
   - Boot disk: Ubuntu 22.04 LTS, 50GB
   - Allow HTTP/HTTPS traffic (optional)

3. SSH into instance (click SSH button in console)

4. Run this script:
   git clone https://github.com/voltavista-lab/symmetrical-parakeet-n0.git
   cd symmetrical-parakeet-n0
   git checkout claude/temporal-fusion-transformers-2u7Wh
   chmod +x deploy_gcp_vm.sh
   ./deploy_gcp_vm.sh

COST SAVING TIP:
----------------
Add --preemptible flag to create command for up to 80% savings:
(Note: Preemptible VMs can be terminated at any time)

gcloud compute instances create tft-training-vm \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --preemptible \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud

EOF
            fi
            ;;
    esac
}

# Run main function
main "$@"
