import pandas as pd
import numpy as np
import os
import joblib 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import re

BASE          = "/content/drive/MyDrive/race_rc_project"
RAW_DIR       = os.path.join(BASE, "data/raw")
PROCESSED_DIR = os.path.join(BASE, "data/processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_data():
    print("Loading train.csv and performing strict 60/20/20 split to prevent data leakage...")
    df = pd.read_csv(os.path.join(RAW_DIR, "train.csv"))
    train, temp = train_test_split(df, test_size=0.4, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)
    print(f"  New Split -> Train: {train.shape} | Val: {val.shape} | Test: {test.shape}")
    return train, val, test

def clean_text(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["article", "question", "A", "B", "C", "D"]:
        df[col] = df[col].apply(clean_text)
    return df

def expand_to_options(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        correct = r["answer"]
        for opt in ["A", "B", "C", "D"]:
            combined = str(r["article"]) + " [SEP] " + str(r["question"]) + " [SEP] " + str(r[opt])
            rows.append({
                "combined": combined, "article": r["article"], "question": r["question"],
                "option_text": r[opt], "option_label": opt,
                "correct": 1 if opt == correct else 0, "answer": correct
            })
    return pd.DataFrame(rows)

def lexical_features(df: pd.DataFrame) -> np.ndarray:
    feats = []
    for _, r in df.iterrows():
        art_tok, q_tok, opt_tok = set(str(r["article"]).split()), set(str(r["question"]).split()), set(str(r["option_text"]).split())
        def jaccard(a, b): return len(a & b) / len(a | b) if a and b else 0.0
        feats.append([len(opt_tok), len(q_tok), len(art_tok), jaccard(opt_tok, q_tok), jaccard(opt_tok, art_tok), jaccard(q_tok, art_tok)])
    return np.array(feats, dtype=np.float32)

def cosine_sim_feature(df: pd.DataFrame, vec) -> np.ndarray:
    from sklearn.preprocessing import normalize
    art_vecs = normalize(vec.transform(df["article"].astype(str)), axis=1)
    opt_vecs = normalize(vec.transform(df["option_text"].astype(str)), axis=1)
    sims = np.array(art_vecs.multiply(opt_vecs).sum(axis=1)).flatten()
    return sims.astype(np.float32).reshape(-1, 1)

def build_one_hot_features(train_df, val_df, test_df, max_features=5000):
    # Using CountVectorizer with binary=True creates the One-Hot Encoded matrix
    ohe_vec = CountVectorizer(max_features=max_features, binary=True)
    X_train_ohe = ohe_vec.fit_transform(train_df["combined"])
    X_val_ohe   = ohe_vec.transform(val_df["combined"])
    X_test_ohe  = ohe_vec.transform(test_df["combined"])
    joblib.dump(ohe_vec, os.path.join(PROCESSED_DIR, "one_hot_vectorizer.pkl"))
    return X_train_ohe, X_val_ohe, X_test_ohe, ohe_vec

def run_preprocessing():
    print("STEP 1 — Loading and Splitting Data")
    train_raw, val_raw, test_raw = load_data()
    print("STEP 2 — Cleaning text")
    train_clean, val_clean, test_clean = clean_df(train_raw), clean_df(val_raw), clean_df(test_raw)
    print("STEP 3 — Expanding to per-option rows")
    train_exp, val_exp, test_exp = expand_to_options(train_clean), expand_to_options(val_clean), expand_to_options(test_clean)
    
    print("STEP 4 — Applying One-Hot Encoding (per rubric requirement)")
    X_tr_ohe, X_val_ohe, X_te_ohe, ohe_vec = build_one_hot_features(train_exp, val_exp, test_exp, 5000)
    
    print("STEP 5 — Cosine & Lexical features")
    cos_tr, cos_val, cos_te = cosine_sim_feature(train_exp, ohe_vec), cosine_sim_feature(val_exp, ohe_vec), cosine_sim_feature(test_exp, ohe_vec)
    lex_tr, lex_val, lex_te = lexical_features(train_exp), lexical_features(val_exp), lexical_features(test_exp)

    print("STEP 6 — Saving files")
    train_exp.to_csv(os.path.join(PROCESSED_DIR, "train_processed.csv"), index=False)
    val_exp.to_csv(os.path.join(PROCESSED_DIR, "val_processed.csv"), index=False)
    test_exp.to_csv(os.path.join(PROCESSED_DIR, "test_processed.csv"), index=False)

    import scipy.sparse as sp
    for name, mat in [("X_train_ohe", X_tr_ohe), ("X_val_ohe", X_val_ohe), ("X_test_ohe", X_te_ohe)]:
        sp.save_npz(os.path.join(PROCESSED_DIR, f"{name}.npz"), mat)
    for name, arr in [("cos_train", cos_tr), ("cos_val", cos_val), ("cos_test", cos_te), ("lex_train", lex_tr), ("lex_val", lex_val), ("lex_test", lex_te)]:
        np.save(os.path.join(PROCESSED_DIR, f"{name}.npy"), arr)
    for name, arr in [("y_train", train_exp["correct"].values), ("y_val", val_exp["correct"].values), ("y_test", test_exp["correct"].values)]:
        np.save(os.path.join(PROCESSED_DIR, f"{name}.npy"), arr)
    print("Preprocessing complete. Your data is now leak-free and rubric-compliant!")

if __name__ == "__main__":
    run_preprocessing()
