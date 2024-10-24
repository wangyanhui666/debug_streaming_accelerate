
## Setup

First, download and set up the repo:

```bash
git clone https://github.com/wangyanhui666/debug_streaming_accelerate.git
cd debug_streaming_accelerate
```

We provide an [`environment.yml`](environment.yml) file that can be used to create a Conda environment. If you only want 
to run pre-trained models locally on CPU, you can remove the `cudatoolkit` and `pytorch-cuda` requirements from the file.

```bash
conda create -n dit python=3.10
conda activate dit

# I use cuda 12.1, pytorch 2.5.0 and 4 A100 gpus
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Traing
### First set config of accelerate in ./config/default_config.yaml
you should set your gpu envirment correctly, like [gpu_ids]....
```yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
dynamo_config:
  dynamo_backend: INDUCTOR
enable_cpu_affinity: false
gpu_ids: all
machine_rank: 0
main_training_function: main
main_process_ip: 127.0.0.1
main_process_port: 12321
mixed_precision: bf16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```
### Second download my streaming datasets
1. Install the `hf_transfer` package:

```bash
pip install hf_transfer
```

2. Enable the Hugging Face transfer feature:

```bash
export HF_HUB_ENABLE_HF_TRANSFER=True
```

3. Download the dataset from Hugging Face:

```bash
huggingface-cli download wangyanhui666/imagenet_vae_mds_fp32 --repo-type dataset --local-dir ./vae_mds_fp32
```

### Third start training
```bash
bash train_local_streamdata.sh
```
