# Multi-Objective Reinforcement Learning from AI Feedback
This repository implements Multi-Objective Reinforcement Learning from AI Feedback (MORLAIF). Instead of training the target model with a single preference model representing "all human preferences", the idea is to break this down into many simpler principles such as "toxicity", "factuality" and "sycophancy". This is essentially a form of task decomposition on the preference modeling stage of a RLAIF or RLHF system. It improves alignment performance, interpretability and allows fine-grained control over the model's behavior without retraining preference models. The results show that MORLAIF outperforms standard single-objective RLAIF baselines, and that it can be used to align larger language models using smaller preference models. For more information read the full paper on arxiv at: [Multi-Objective Reinforcement Learning from AI Feedback](https://arxiv.org/abs/2406.07295).
## Table of Contents
- [Replication](#replication)
- [Methodology](#methodology)
- [Current Setup](#current-setup)
- [Principles](#principles)
- [Results](#results)
- [Theoretical Advantages](#theoretical-advantages)
- [Frequently Asked Questions (FAQ)](#frequently-asked-questions-faq)
- [Acknowledgments](#acknowledgments)

## Replication:
1. **Build Docker Image**: Run `docker build -t morlaif .` to build the Docker environment.
2. **Run Docker Container**: Use `docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm morlaif` to start the container.
3. **Prepare the Dataset**: Execute `process_HH_dataset.py` to format the anthropic `hh-rlhf` dataset. The principle specific datasets in the `data/datasets` folder can alternatively be used.  
4. **Acquire Feedback**: Run `get_feedback.bash` to obtain feedback from GPT 3.5 or 4 according to the different principles in `principles`. Note that this will by defualt send API requests in parallel, set `max_requests_per_minute` appropriately. There is also the option of using OpenAI's batch api using `create_batches.py` and `send_batches.py`
5. **Launch PM Training**: Initiate preference model training with the provided bash scripts in `PM_training`.
6. **Create a Scalarization Function**: A few different scalarization functions (see scalarization section) are defined in `PPO_training/MORL_scalarizer.py`. Good weights for linear can be calculated using `PM_training/PM_regression.py`
7. **Start PPO Training**: Follow up with PPO training using the designated bash scripts in `PPO_training`.

## Methodology:

 ![](https://github.com/carolius/MORLAIF/blob/main/MORLAIF.png?raw=true)
 
### Preference modeling:
1.	**Generating responses:** A SFT model produces pairs of responses for prompts. 
2.	**Rating by Feedback Model:** A feedback model evaluates which of these responses is better according to each individual principle. Most experiments used the 12 principles listed below.
3.	**Training Preference Models:** The ratings are used to train separate preference models (full models or LoRAs) for each principle.

### RL from AI feedback:

4.	**MORL Scalarization Function:** A MORL scalarization function combines the ratings from each preference model into a reward signal.
5.	**PPO Training:** The combined score from the scalarization function acts as a reward signal, guiding the training of the target model using Proximal Policy Optimization (PPO).

## Current Setup:
- **Target Model:** The code currently supports GPT-2 small/medium/large/XL, Llama-2-7B/13B/70B and Gemma-2B/7B.
- **Preference Models:** Currently the code implements the finetuning of GPT-2, Llama-2, Gemma or LoRAs of these models as the preference models.
- **Feedback Model:** GPT-3.5 and GPT-4 are supported to rate response pairs according to each individual principle.
- **Datasets:** Currently Anthropic's hh-rlhf and openassistant-guanaco are used.
- **Hardware:** For GPT2 models a RTX 3090 24GB was used while for Llama and Gemma models a remote cluster with 8x A100 80gb was used.
## Principles:
These 12 principles were used for most of the experiments. 
1. helpfulness
2. ethicality
3. factuality
4. toxicity
5. sycophancy
6. empathy
7. relevance
8. context
9. bias
10. understandability
11. detail
12. conciseness
## MORL Scalarization Functions
Different MORL scalarization functions were evaluated to combine the preference model outputs, including:
- Weighted Linear Combination
- Worst-Case Optimization, aka Minimax, Max-Min or Rawlsian social welfare
- Soft Max-Min
- Uncertainty-Weighted Optimization
- Lower Quantile Optimization
- Max-Median
- Bernoulli-Nash
## Results
<p align="center">
  <img src="https://github.com/carolius/Multi-Objective-Reinforcement-Learning-from-AI-Feedback/blob/main/plots/principle_accuracy.png" alt="Image 1" width="45%">
  <img src="https://github.com/carolius/Multi-Objective-Reinforcement-Learning-from-AI-Feedback/blob/main/plots/objective_accuracy.png" alt="Image 2" width="45%">
</p>

Our experiments demonstrate that when trained as preference models for individual principles, the accuracies are generally much higher than for single-objective PMs. Furthermore, all MORL objectives outperform the standard single-PM RLAIF baselines.

<p align="center">
  <img src="https://github.com/carolius/Multi-Objective-Reinforcement-Learning-from-AI-Feedback/blob/main/plots/human_winrate.png" alt="Image 3" width="45%">
  <img src="https://github.com/carolius/Multi-Objective-Reinforcement-Learning-from-AI-Feedback/blob/main/plots/LLM_winrate.png" alt="Image 4" width="45%">
</p>
In human preference experiments, MORLAIF Llama-2-7b is strongly preferred over Single Objective RLAIF. Notably, a version trained with GPT-2-medium preference models performs on par with the single-objective model. GPT-4-Turbo judgments show high win rates for GPT-2-medium with a decreasing but still significant win rate for larger models.

<p align="center">
  <img src="https://github.com/carolius/Multi-Objective-Reinforcement-Learning-from-AI-Feedback/blob/main/plots/principle_correlations.png" alt="Image 5" width="45%">
  <img src="https://github.com/carolius/Multi-Objective-Reinforcement-Learning-from-AI-Feedback/blob/main/plots/principle_ablation.png" alt="Image 6" width="45%">
</p>
The correlation matrix reveals weak correlations for sycophancy, which is also the only principle which received a negative weight, indicating that sycophancy is actually preferred. Multi-objective PM accuracy depends on the number of principles used, shown for GPT-2-medium, Llama-2-7b, and the theoretical performance ceiling (representing 100\% accuracy for each principle).

## Acknowledgments
- Special thanks to the Long-Term Future Fund for funding this project.