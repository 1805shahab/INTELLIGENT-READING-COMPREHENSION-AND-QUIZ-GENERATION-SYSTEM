sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(2, 2, figsize=(16, 10))

# Chart 1: Answer Balance (Checks for bias in A, B, C, D)
sns.countplot(x='answer', data=df, order=['A', 'B', 'C', 'D'], ax=ax[0, 0], palette='viridis')
ax[0, 0].set_title("Correct Answer Distribution (Balance Check)")
ax[0, 0].set_ylabel("Count")

# Chart 2: Passage Length Distribution
sns.histplot(df['a_len'], bins=40, ax=ax[0, 1], color='coral', kde=True)
ax[0, 1].set_title("Distribution of Article Lengths")
ax[0, 1].set_xlabel("Word Count")

# Chart 3: Question Length Distribution
sns.histplot(df['q_len'], bins=30, ax=ax[1, 0], color='skyblue', kde=True)
ax[1, 0].set_title("Distribution of Question Lengths")
ax[1, 0].set_xlabel("Word Count")

# Chart 4: Question Types (Who, What, Where, etc.)
def get_q_type(q):
    q = str(q).lower()
    for w in ['what', 'why', 'how', 'which', 'who', 'where', 'when']:
        if w in q: return w
    return 'other/fill-in'

df['q_type'] = df['question'].apply(get_q_type)
sns.countplot(x='q_type', data=df, order=['what', 'why', 'how', 'which', 'who', 'where', 'when', 'other/fill-in'], ax=ax[1, 1], palette='magma')
ax[1, 1].set_title("Question Type Distribution")
ax[1, 1].set_ylabel("Count")

plt.tight_layout()
plt.show()
