import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Setup paths and load data
b_dir = "/content/drive/MyDrive/race_rc_project"
p_dir = os.path.join(b_dir, "data/processed")

# Load the validation set (or train set if available)
df = pd.read_csv(os.path.join(p_dir, "val_processed.csv"))

print("--- Data Overview ---")
print(f"Total Rows: {len(df)}")
print(f"Columns: {list(df.columns)}\n")

# 2. Calculate text lengths (words)
df['a_len'] = df['article'].astype(str).apply(lambda x: len(x.split()))
df['q_len'] = df['question'].astype(str).apply(lambda x: len(x.split()))

# Calculate average option length
def opt_len(r):
    opts = [str(r.get(c, '')) for c in ['A', 'B', 'C', 'D'] if pd.notnull(r.get(c))]
    return sum(len(o.split()) for o in opts) / 4 if opts else 0

df['o_len'] = df.apply(opt_len, axis=1)

# 3. Summary Statistics Table
print("--- Summary Statistics Table ---")
stats = df[['a_len', 'q_len', 'o_len']].describe().round(2)
stats.columns = ['Article Length', 'Question Length', 'Avg Option Length']
print(stats)
