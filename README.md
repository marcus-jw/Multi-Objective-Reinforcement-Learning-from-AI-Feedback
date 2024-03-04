# Multi-Objective Reinforcement Learning from AI Feedback (WIP)
This repository implements Multi-Objective Reinforcement Learning from AI Feedback (MORLAIF). Instead of training the target model with a single preference model representing "human preferences" we break this down into many simpler principles such as "toxicity”, “truthfulness” and “sycophancy”. This is essentially a form of task decomposition on the preference modeling stage of a RLAIF or RLHF system. It seems likely that this will improve alignment performance, reduce Goodharting and improve interpretability. This project aims to investigate and measure this.

## Table of Contents
- [How to Replicate](#how-to-replicate)
- [Process](#process)
- [Current Setup](#current-setup)
- [Preliminary Results](#preliminary-results)
- [Advantages](#advantages)
- [Frequently Asked Questions (FAQ)](#frequently-asked-questions-faq)
- [Acknowledgments](#acknowledgments)

## How to replicate:
### Requirements
- Docker
- OpenAI API Key

### Steps
1. **Build Docker Image**: Run `docker build -t morlaif .` to build the Docker environment.
2. **Run Docker Container**: Use `docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm morlaif` to start the container.
3. **Prepare the Dataset**: Execute `process_HH_dataset.py` to format the anthropic `HH-rlhf` dataset.
4. **Generate Responses**: Use `generate_responses_GPT2.py` or `generate_responses_llama.py`  for sampling from the target model.
5. **Acquire Feedback**: Run `get_feedback.bash` to obtain feedback from GPT 3.5 or 4 (OpenAI API key required). Note that this will send API requests in parallel set max_requests_per_minute appropriately.
6. **Launch PM Training**: Initiate preference model training with the provided bash scripts.
7. **Start PPO Training**: Follow up with PPO training using the designated bash scripts.

## Process:

 ![](https://github.com/carolius/MORLAIF/blob/main/MORLAIF.png?raw=true)
 
### Preference modeling:
1.	**Sampling from Target Model:** The target model produces pairs of responses for prompts. This is implemented in `generate_responses_GPT2.py` which uses the HH-rlhf dataset.
2.	**Rating by Feedback Model:** A feedback model evaluates which of these responses is better for each individual principle.
3.	**Training Preference Models:** These ratings are then used to train a separate preference model for each principle. The preference model can either be a full model or a Low Rank Adaptation (LoRA).

### RL from AI feedback:

4.	**MORL Scalarization Function:** Each preference model will assign a rating to a given output, these scores are then combined using a scalarization function. This function can be anything from a simple weighted sum to more complicated functions such as max-min or lexicographic priorities. A few scalarization functions are implemented in `MORL_scalarizer.py`. 
5.	**PPO Training:** The combined score from the scalarization function acts as a reward signal, guiding the training of the target model using Proximal Policy Optimization (PPO).

## Current Setup:
- **Target Model:** The code currently supports GPT-2 medium, large and XL along with Llama-7B. I plan to add Llama-13B, Llama-70B, Gemma-2B and Gemma-7B.
- **Preference Models:** Currently the code implements the finetuning of GPT-2, Llama-7B or LoRAs of these models as the preference models. Again, I plan to add more Llama models and Gemma.
- **Feedback Model:** GPT-3.5 and GPT-4 are used to rate response pairs according to each individual principle.
- **Dataset:** Currently Anthropic's HH-rlhf dataset is used; more datasets will be added in the future.
- **Hardware:** For GPT2 models a RTX 3090 24GB was used while for Llama models an instance with 8x A100 80gb was used
## Preliminary Results
 ![](https://github.com/carolius/MORLAIF/blob/main/results.png?raw=true)
For the larger models, using a weighted sum of the preference models seems to do significantly better than standard Constitutional AI. Both softmin and setting the sycophancy weight to zero seems to degrade performance to roughly inline with single-objective. That setting the sycophancy PM weight to zero reduces performance makes sense as if humans prefer sycophancy reducing it should align worse with human preferences, therefore even though zero-syco performs worse than linear it might make more sense to use it.
## Advantages

**More specific and unique principles.** Unlike Anthropic's Constitutional AI principles which are quite general, contain many different tasks and are similar to each other; MORLAIF principles could be made very specific and unique.  For example: “Please choose the assistant response that is as harmless and ethical as possible. Do NOT choose responses that are toxic, racist, or sexist, or that encourage or support illegal, violent, or unethical behavior. Above all the assistant’s response should be wise, peaceful, and ethical” could be turned into separate principles for toxicity, violence, illegality etc. It seems likely that it is an easier task to determine whether a response is one of these things than all of them together. This means that labelling will likely be better, leading to improved safety performance of the final model.

**More principles.** MORLAIF allows us to include numerous minor principles without diluting focus on major ones. This can be done by giving these lesser principles low weight or only optimizing them after the important principles reach a certain threshold.

**More interpretable system.** MORLAIF systems are inherently more interpretable as each reward function is more specific and you can see how a response would score on the different reward functions.

**Easier to fine-tune.** In a MORLAIF system you can change the behavior of the target model without needing to retrain the preference models, and the output space is in some sense continuous allowing you to easily reach any point on a Pareto frontier. With standard constitutional AI the weighting of the principles in a constitution is implicit and depends on the wording, number of principles which contain the thing you care about. Say you have a well-trained model that, apart from occasionally outputting violent content, performs well. To fix the violence problem should you A) add another principle which targets violence more specifically, B) add violence as a consideration to a larger proportion of your principles or C) reword your principles to put a larger focus on violence? In MORL by contrast, you could simply increase the weight of your violence reward function.

**Less Goodharting/overfitting/reward hacking.** Training on multiple objectives means that we are less likely to end up with some extreme Goodharted solution as our model must score well on all objectives. “Human preferences” are quite vague whereas a single principle can be made very specific, meaning that a model trained on many of them will be less likely to overfit even with extreme optimization pressure. Reward hacking is a large problem when training using preference models, which is why most methods use some regularization such as KL-divergence to make sure the model doesn't stray too far from the original weights. I hypothesize that in the multi-objective case we will require much less regularization.

**Less risk of deception in the preference model.** Since preference models can be made much smaller than they are now it seems likely that it will be harder for them to be deceptive and easier for us to detect it if they are. This is a form of task decomposition; we decompose the task of the preference model into many sub tasks which each represent a simpler component of the full task of representing human preferences. 



## Frequently Asked Questions (FAQ)
**Q: Will this cause optimization issues since the rewards aren't Markovian?**

**A:** For linear combinations of different principles, there are no concerns, since then the multi-objective MDP can always be transformed into a single objective MDP, which ensures that convergence proofs remain applicable. A more complex MORL scalarization function would make methods such as Q-learning inapplicable but policy gradient methods like PPO should still work. For our purposes the MORL scalarization function being monotonic in each reward is sufficient for trainability which seems like a reasonable property of the function to have anyway. In other words, we can't optimize for a certain level of, for example, toxicity; it must always be preferable to reduce toxicity, all other reward functions being equal.

**Q: Doesn’t this need a lot of extra compute?**

**A:** More compute, yes, a lot more? Probably not. I would argue that training preference models is a fairly cheap step compared to training base models. It seems likely that understanding one principle at a time is a simpler task than all at once, so the preference models could likely be made smaller and might train faster than for the single objective case. I would further argue that the compute needed for an end product/what you see in a paper is not the only compute required. In a single objective system, you likely need to retrain the preference model multiple times to experiment with different principles, their wordings and their number. Switching to MORLAIF would save you the need to retrain the preference models to change the reward signal.  

**Q: Doesn’t this require a lot of extra memory during RL training?**

**A:** Potentially, depending on how it is done. As previously mentioned, preference models might be made smaller using this scheme. In one of the current implementations, all the preference models are different LoRAs linked to the same base model. This approach results in minimal additional memory consumption, as the preference models (PMs) use the same base model, with different LoRAs hot-swapped in.

**Q: How will you choose principle weights and the MORL scalarization function?**

**A:** Figuring this out is a large part of the project. Models with different weights and scalarization functions will be trained and analyzed for their performance while iteratively making improvements. As a starting point doing some form of regression on a human preference dataset should be a way to get some strong initial weights.

## Acknowledgments
- Special thanks to the Long-Term Future Fund for funding this project.