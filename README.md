#Multi-Objective Reinforcement Learning from AI Feedback (WIP)
This repository implements Multi-Objective Reinforcement Learning from AI Feedback  (MORLAIF) using Torch, Huggingface Transformers and TRL. 
#Current Setup:
-**Target Model:** The code currently employs GPT-2-medium for testing purposes, but the goal is to upgrade to llama-7b.
-**Preference Models:** The code currently implements both using GPT-2-medium for preference models and using different LoRAs to GPT-2 for the different PMs. The plan is to test training llama-7b models as PMs, training LoRAs of llama-7b and to try using smaller models.
-**Feedback Model:** GPT-3.5-Turbo serves as the foundation model which rates response pairs according to each individual principle.
-**Dataset:** Currently Anthropic’s HH-rlhf dataset is used although more may be added in the future.
#Process Flow:

 ![](https://github.com/carolius/MORLAIF/blob/main/MORLAIF.png?raw=true)
 
**Preference modeling:**
1.	**Sampling from Target Model:** The target model produces pairs of responses for prompts. This is implemented in generate_responses_GPT2.py which uses the HH-rlhf dataset.
2.	**Rating by Feedback Model:** A foundation model evaluates which of these responses is better for each Individual principle. This is implemented in get_feedback_from_GPT-3.5.py.
3.	**Training Preference Models:** These ratings are then used to train a separate preference model for each principle. This is implemented in train_preference_model.py and train_preference_model_LoRA.py.
**RL from AI feedback:**
4.	**MORL Scalarization Function:** Each preference model will assign a rating to a given output, these scores are then combined using a scalarization function. This function can be anything from a simple weighted sum to more complicated functions such as max-min or lexicographic priorities. A few scalarization functions are implemented in MORL_scalarizer.py. 
5.	**PPO Training:** The combined score from the scalarization function acts as a reward signal, guiding the training of the target model using Proximal Policy Optimization (PPO). This is implemented in PPO_RL_training.py.

