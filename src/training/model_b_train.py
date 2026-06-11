import os, re, numpy as np, pandas as pd, joblib
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MaxAbsScaler, normalize

BASE = "/content/drive/MyDrive/race_rc_project"
PROCESSED_DIR = os.path.join(BASE, "data/processed")
MODEL_B_DIR   = os.path.join(BASE, "models/model_b/traditional")
STOPWORDS = set("a an the is are was were be been being have has had do does did will would could should may might shall can in on at of for to with and or but not i you he she it we they this that these those from by about into".split())

def clean(text): return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", str(text).lower())).strip()
def tokenize(text): return str(text).lower().split()
def jaccard(a, b): return len(set(a) & set(b)) / len(set(a) | set(b)) if set(a) and set(b) else 0.0

def build_distractor_dataset(df, vec, sample_n=None):
    wide = df.drop_duplicates(subset=["article","question","answer","option_label"]).pivot(index=["article","question","answer"], columns="option_label", values="option_text").reset_index()
    wide.columns.name = None
    for c in ["A","B","C","D"]:
        if c not in wide.columns: wide[c] = ""
    unique_q = wide.dropna(subset=["article","question"])
    
    if sample_n is not None and len(unique_q) > sample_n: 
        unique_q = unique_q.sample(sample_n, random_state=42)
    
    all_feats, all_labels = [], []
    for _, r in unique_q.iterrows():
        article, correct_text = str(r["article"]), str(r.get(r["answer"], ""))
        wrong_tokens = set()
        for o in ["A","B","C","D"]:
            if o != r["answer"]: wrong_tokens.update(tokenize(str(r.get(o, ""))))
        tokens = tokenize(article)
        cands = [w for w, _ in Counter([t for t in tokens if t not in STOPWORDS and len(t)>2]).most_common(15) if w != correct_text]
        if not cands: continue
        cand_vecs, ans_vec = normalize(vec.transform(cands), axis=1), normalize(vec.transform([correct_text]), axis=1)
        cos_sims = np.array(cand_vecs.multiply(ans_vec).sum(axis=1)).flatten()
        for i, cand in enumerate(cands):
            all_feats.append([cos_sims[i], len(set(cand) & set(correct_text)) / max(len(set(cand) | set(correct_text)), 1), tokens.count(cand)/max(len(tokens),1), len(cand.split()), 0.0])
            all_labels.append(1 if cand in wrong_tokens else 0)
    return np.array(all_feats, dtype=np.float32), np.array(all_labels, dtype=int)

def build_hint_dataset(df, sample_n=None):
    wide = df.drop_duplicates(subset=["article","question","answer","option_label"]).pivot(index=["article","question","answer"], columns="option_label", values="option_text").reset_index()
    wide.columns.name = None
    for c in ["A","B","C","D"]:
        if c not in wide.columns: wide[c] = ""
    unique_q = wide.dropna(subset=["article","question"])
    
    if sample_n is not None and len(unique_q) > sample_n: 
        unique_q = unique_q.sample(sample_n, random_state=42)
        
    rows_X, rows_cls = [], []
    for _, r in unique_q.iterrows():
        article, question = clean(str(r["article"])), clean(str(r["question"]))
        ans_toks = set(tokenize(clean(str(r.get(str(r.get("answer", "")).strip(), "")))))
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', article) if len(s.split()) > 3]
        for pos, sent in enumerate(sents):
            s_tok = set(tokenize(sent))
            rows_X.append([jaccard(s_tok, set(t for t in tokenize(question) if t not in STOPWORDS)), pos / max(len(sents) - 1, 1), min(len(sent.split()) / 50.0, 1.0), jaccard(s_tok, set(t for t in tokenize(article) if t not in STOPWORDS))])
            rows_cls.append(1 if len(s_tok & ans_toks) / max(len(ans_toks), 1) > 0 else 0)
    return np.array(rows_X, dtype=np.float32), np.array(rows_cls, dtype=int)

def run_model_b():
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train_processed.csv"))
    
    # 🚨 RUBRIC UPDATE: Load the new One-Hot Vectorizer instead of BoW
    vec = joblib.load(os.path.join(PROCESSED_DIR, "one_hot_vectorizer.pkl"))

    print("\n[1/2] Building Distractors (FULL DATASET)...")
    Xd_tr, yd_tr = build_distractor_dataset(train_df, vec, sample_n=None)  
    scaler_d = MaxAbsScaler()
    lr_d = LogisticRegression(class_weight="balanced", max_iter=300).fit(scaler_d.fit_transform(Xd_tr), yd_tr)
    joblib.dump(lr_d, os.path.join(MODEL_B_DIR, "distractor_lr.pkl"))
    joblib.dump(scaler_d, os.path.join(MODEL_B_DIR, "distractor_scaler.pkl"))
    
    rf_d = RandomForestClassifier(n_estimators=50, class_weight="balanced", n_jobs=-1, random_state=42).fit(Xd_tr, yd_tr)
    joblib.dump(rf_d, os.path.join(MODEL_B_DIR, "distractor_rf.pkl"))
    
    print("[2/2] Building Hints (FULL DATASET)...")
    Xh_tr, yh_cls_tr = build_hint_dataset(train_df, sample_n=None) 
    scaler_h = MaxAbsScaler()
    lr_h = LogisticRegression(class_weight="balanced", max_iter=300).fit(scaler_h.fit_transform(Xh_tr), yh_cls_tr)
    joblib.dump(lr_h, os.path.join(MODEL_B_DIR, "hint_clf.pkl"))
    joblib.dump(scaler_h, os.path.join(MODEL_B_DIR, "hint_scaler.pkl"))
    print("\nModel B Training Complete! All .pkl files generated.")

if __name__ == "__main__":
    run_model_b()
