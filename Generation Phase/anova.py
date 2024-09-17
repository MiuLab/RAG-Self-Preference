import json
import numpy as np
from scipy.stats import f_oneway

with open('results/nq_perplexity.json', 'r') as f:
  data = json.load(f)

# Perform one-way ANOVA
f_stat, p_value = f_oneway(data['human'], data['gpt'], data['llama'])

# Display results
print(f"F-statistic: {f_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpretation
if p_value < 0.05:
    print("There is a statistically significant difference between the groups.")
else:
    print("No statistically significant difference between the groups.")



# SEE WHICH GROUPS ARE DIFFERENT
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Combine the data into a pandas DataFrame for Tukey's test
data_combined = pd.DataFrame({
    'score': data['human'] + data['gpt'] + data['llama'],
    'group': ['human']*1000 + ['gpt']*1000 + ['llama']*1000
})

# Perform Tukey's HSD post-hoc test
tukey = pairwise_tukeyhsd(endog=data_combined['score'], groups=data_combined['group'], alpha=0.05)

# Display results
print(tukey)