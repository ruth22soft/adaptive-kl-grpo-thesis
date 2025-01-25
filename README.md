# simpleRL-reason
This is a replicate of DeepSeek-R1-Zero training and DeepSeek-R1 training on small models with limited data

## Quick Start

### Installation

Our code is implemented based on [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF/tree/main?tab=readme-ov-file#installation). Please follow OpenRLHF's guidance to configure other environments and install our version:

```bash
git clone https://github.com/hkust-nlp/simpleRL-reason.git
cd train
pip install -e .
```

### Reproducing SimpleRL-Zero

We use PPO with Ray and vLLM acceleration for training:

```bash
# launch the master node of ray in container
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

# if you want to launch ray on more nodes, use
ray start --address {MASTER-NODE-ADDRESS}:6379  --num-gpus 8


# Submit ray task on the master node


cd train/examples/script

ray job submit --address="http://127.0.0.1:8265" \
        --runtime-env-json='{
        "pip": ["ray==2.12.0", "latex2sympy2", "timeout_decorator"]
    }' -- /bin/bash train_ppo_qwen_base_math_lv35_new.sh
```

