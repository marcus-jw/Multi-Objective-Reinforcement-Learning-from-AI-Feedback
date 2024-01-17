import matplotlib.pyplot as plt
import json
import pandas as pd
import seaborn as sns
# Read the JSONL file
results = []
with open('accuracy_results.jsonl', 'r') as file:
    for line in file:
        results.append(json.loads(line))
df = pd.DataFrame(results)

sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
ax = sns.lineplot(x="model", y="accuracy", hue="Objective function", data=df)
ax.set_ylim(0.5, 1)  # Set y-axis range from 0.5 to 1
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.tight_layout() 
plt.savefig('results.png', dpi=300)
plt.show()

