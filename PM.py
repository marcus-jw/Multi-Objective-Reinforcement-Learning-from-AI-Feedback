
    
    import csv
import torch

class RewardModel:
    def __init__(self, filename):
        self.data = self._load_data(filename)

    def _load_data(self, filename):
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # skip header
            return {row[0]: {'answer1': row[1], 'logits_answer1': float(row[2]),
                             'answer2': row[3], 'logits_answer2': float(row[4])} for row in reader}

    def get_reward(self, question, response):
        if question not in self.data:
            return 0.0  # or some default value
        
        answer_data = self.data[question]
        
        if response == answer_data['answer1']:
            return answer_data['logits_answer1']
        elif response == answer_data['answer2']:
            return answer_data['logits_answer2']
        else:
            return 0.0  # or some default value if the response doesn't match either answer

    def __call__(self, responses, questions):
        rewards = [self.get_reward(q, r) for q, r in zip(questions, responses)]
        return torch.tensor(rewards)