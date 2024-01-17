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
ax = sns.lineplot(x="model", y="accuracy", hue="Objective function", data=df, linewidth=2.5)
ax.set_ylim(0.5, 1) 
plt.title("Accuracy of the models when trained with different objective functions", fontsize=20)
plt.xlabel("Model", fontsize=18)  
plt.ylabel("Accuracy", fontsize=18)  
plt.xticks(fontsize=15)  
plt.yticks(fontsize=15)  
plt.tight_layout() 

ax.legend(fontsize=16)

plt.savefig('results.png', dpi=300)
plt.show()
