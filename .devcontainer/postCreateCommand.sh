git config --global safe.directory '*'
git config --global core.editor "code --wait"
git config --global pager.branch false

# Activate conda by default
echo "source /home/vscode/miniconda3/bin/activate" >> ~/.zshrc
echo "source /home/vscode/miniconda3/bin/activate" >> ~/.bashrc

# Use gpt4roi environment by default
echo "conda activate gpt4roi" >> ~/.zshrc
echo "conda activate gpt4roi" >> ~/.bashrc

# Activate conda on current shell
source /home/vscode/miniconda3/bin/activate

# Create and activate gpt4roi environment
conda create -y -n gpt4roi python=3.9
conda activate gpt4roi

echo "Installing CUDA..."
# Even though cuda package installs cuda-nvcc, it doesn't pin the same version so we explicitly set both
conda install -y -c nvidia cuda=11.7 cuda-nvcc=11.7

export CUDA_HOME=/home/vscode/miniconda3/envs/gpt4roi
echo "export CUDA_HOME=$CUDA_HOME" >> ~/.zshrc
echo "export CUDA_HOME=$CUDA_HOME" >> ~/.bashrc

pip install --upgrade pip
pip install setuptools_scm
pip install --no-cache-dir -e .

# please use conda re-install the torch, pip may loss some runtime lib
conda install -y -c nvidia -c pytorch pytorch-cuda=11.7 pytorch=1.10.0 torchvision=0.11.1 torchaudio=0.10.0

pip install ninja
pip install flash-attn --no-build-isolation

MMCV_WITH_OPS=1 pip install -e mmcv-1.4.7

git clone https://huggingface.co/jeffwan/llama-7b-hf ./llama-7b
git clone https://huggingface.co/shilongz/GPT4RoI-7B-delta-V0 ./GPT4RoI-7B-delta

export PYTHONPATH=`pwd`:$PYTHONPATH
python3 -m scripts.apply_delta \
    --base ./llama-7b \
    --target ./GPT4RoI-7B \
    --delta ./GPT4RoI-7B-delta

curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.3/install.sh | bash

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion

nvm install 18.16.0

curl -fsSL https://get.pnpm.io/install.sh | sh -

export PNPM_HOME="/home/vscode/.local/share/pnpm"
case ":$PATH:" in
  *":$PNPM_HOME:"*) ;;
  *) export PATH="$PNPM_HOME:$PATH" ;;
esac

source ~/.zshrc

pnpm --version

git clone https://github.com/ShoufaChen/gradio-dev.git
cd gradio-dev
bash scripts/build_frontend.sh
pip install -e .
cd ..

node -v
npm -v
nvcc -V
TORCH_VERSION=$(python -c "import torch;print(torch.__version__)")
echo "Torch version: $TORCH_VERSION"

echo "postCreateCommand.sh completed!"
