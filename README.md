

<div align="center">

# 7B Model and 8K Examples: Emerging Reasoning with Reinforcement Learning is Both Effective and Efficient

[![Notion](https://img.shields.io/badge/Notion-%23000000.svg?style=for-the-badge&logo=notion&logoColor=white)](https://hkust-nlp.notion.site/simplerl-reason) 

</div>


![simplelr-reaoning-intro-figure_00](https://github.com/user-attachments/assets/bacd1680-ccb0-4921-a687-8a595ebf5896)

> Training dynamics of our Qwen2.5-SimpleRL-Zero training starting from the Qwen2.5-Math-7B, without SFT or reward models.



Many researchers are exploring possible paths towards learning o-style models, such as distillation, MCTS, process-based reward models, and reinforcement learning. Recently, both [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) and [Kimi-k1.5](https://github.com/MoonshotAI/Kimi-k1.5) demonstrate an extremely simple recipe on this path, using simple RL algorithms to learn emerging long CoT and self-reflection patterns and leading to strong results, where no MCTS and reward models are used. However, their experiments are based on  huge models in a large-scale RL setting. It remains unknown whether small models can demonstrate similar behaviors, how much data is needed, and how would the quantitative results compare with other approaches. We reproduce the training of DeepSeek-R1-Zero and DeepSeek-R1 for complex mathematical reasoning, starting from Qwen-2.5-Math-7B (base model), and only using 8K (query, final answer) examples from the original MATH dataset. We are surprised how far the 8K MATH examples lift this 7B base model without any other external signals:

***All results are in pass@1 accuracy***


|                            | AIME 2024 | MATH 500 | AMC  | Minerva Math | OlympiadBench | Avg.  |
|---------------------------------|-----------|----------|------|--------------|---------------|-------|
| Qwen2.5-Math-7B-Base            | 16.7      | 52.4     | 52.5 | 12.9         | 16.4          | 30.2  |
| Qwen2.5-Math-7B-Base + 8K MATH SFT | 3.3       | 54.6     | 22.5 | 32.7         | 19.6          | 26.5  |
| Qwen-2.5-Math-7B-Instruct       | 13.3      | 79.8     | 50.6 | 34.6         | 40.7          | 43.8  |
| Llama-3.1-70B-Instruct          | 16.73     | 64.6     | 30.1 | 35.3         | 31.9          | 35.7  |
| rStar-Math-7B                   | 26.7      | 78.4     | 47.5 | -            | 47.1          | -     |
| Eurus-2-7B-PRIME                | 26.7      | 79.2     | 57.8 | 38.6         | 42.1          | 48.9  |
| Qwen2.5-7B-SimpleRL-Zero        | 33.3      | 77.2     | 62.5 | 33.5         | 37.6          | 48.8  |
| Qwen2.5-7B-SimpleRL             | 26.7      | 82.4     | 62.5 | 39.7         | 43.3          | 50.9  |

Qwen2.5-7B-SimpleRL-Zero is the simple RL training from the base model directly, using only 8K MATH examples. It achieves gains of nearly 20 absolute points on average compared to the base model. Moreover, it outperforms Qwen-2.5-Math-7B-Instruct on average, and is roughly comparable to the recently released [Eurus-2-7B-PRIME](https://github.com/PRIME-RL/PRIME) and [rStar-Math-7B](https://arxiv.org/abs/2501.04519) which are also based on Qwen-2.5-Math-7B. These baselines contain much more complicated components such as reward models and use at least 50x more and advanced data:

***Data comparison of different approaches***


|                   | Qwen2.5-Math-7B-Instruct | rStar-Math-7B | Eurus-2-7B-PRIME | Qwen2.5-7B-SimpleRL-Zero |
|---------------------------|--------------------------|---------------|------------------|--------------------------|
| **Base Model**            | Qwen2.5-Math-7B         | Qwen2.5-Math-7B | Qwen2.5-Math-7B  | Qwen2.5-Math-7B          |
| **SFT Data**              | 2.5M (open-source and in-house) | ~7.3M (MATH, NuminaMath, etc.) | 230K | 0 |
| **RM Data**               | 618K (in-house)         | ~7k (in-house) | 0                | 0                        |
| **RM**                    | Qwen2.5-Math-RM (72B)   | None          | Eurus-2-7B-SFT   | None                     |
| **RL Data**               | 66K queries × 32 samples | ~3.647M × 16  | 150K queries × 4 samples | 8K queries × 8 samples |

We are both excited and surprised by the significant gains achieved using only 8K MATH examples. Notably, while the MATH queries are considerably easier than many challenging benchmarks such as AIME and AMC, this simple RL recipe demonstrates remarkable generalization, with performance increasing by at least 10 absolute points compared to the base model. This easy-to-hard generalization effect is something we could not have envisioned with standard SFT training on the same dataset. We fully open-source our training code and details, hopefully as a strong baseline setup for the community to further explore the potential of RL for reasoning.


## News

- **[2025/01/25]** We release the training/eval code and our blog. We are working on the paper and will release it very soon.

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

### Reproducing SimpleRL

Comming Soon！


### Evaluate

We used [Qwen Math's codebase](https://github.com/QwenLM/Qwen2.5-Math/tree/main/evaluation) for evaluation, but for fairness considerations, we completely prohibited solving problems by calling code. Please follow the `/eval` instructions for evaluation


## Citation

```bibtex
@misc{zeng2025emerging,
  title={7B Model and 8K Examples: Emerging Reasoning with Reinforcement Learning is Both Effective and Efficient},
  author={Weihao Zeng and Yuzhen Huang and Wei Liu and Keqing He and Qian Liu and Zejun Ma and Junxian He},
  year={2025},
  howpublished={\url{https://hkust-nlp.notion.site/simplerl-reason}},
  note={Notion Blog}
}
```


## Acknowledgement
We implement our reinforcement learning algorithm extending from [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF). We utilize [vLLM](https://github.com/vllm-project/vllm) for inference and develop evaluation scripts based on [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math/tree/main/evaluation). Particularly, we thank the developers of DeepSeek-R1 and Kimi-k1.5 for their innovation and contribution to the open-source community.

## Starchart
[![Star History Chart](https://api.star-history.com/svg?repos=hkust-nlp/simpleRL-reason&type=Date)](https://star-history.com/#hkust-nlp/simpleRL-reason&Date)



