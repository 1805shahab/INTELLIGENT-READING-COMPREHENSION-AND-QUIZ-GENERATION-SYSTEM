import os, time, joblib, re, numpy as np, pandas as pd
import scipy.sparse as sp
from collections import Counter
from sklearn.preprocessing import normalize

BASE = "/content/drive/MyDrive/race_rc_project"
PROCESSED_DIR = os.path.join(BASE, "data/processed")
MODEL_A_DIR = os.path.join(BASE, "models/model_a/traditional")
MODEL_B_DIR = os.path.join(BASE, "models/model_b/traditional")
STOPWORDS = set("a an the is are was were be been being have has had do does did will would could should may might shall can in on at of for to with and or but not i you he she it we they this that these those from by about into".split())

_cache = {}
def _load(path):
    if path not in _cache: _cache[path] = joblib.load(path)
    return _cache[path]

def clean(text): return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", str(text).lower())).strip()
def tokenize(text): return str(text).lower().split()
def jaccard(a, b): return len(set(a) & set(b)) / len(set(a) | set(b)) if set(a) and set(b) else 0.0

def verify_answer(article, question, option_text, model_name="lr"):
    start_time = time.time()
    vec = _load(os.path.join(PROCESSED_DIR, "one_hot_vectorizer.pkl"))
    scaler = _load(os.path.join(MODEL_A_DIR, "model_a_scaler.pkl"))
    model = _load(os.path.join(MODEL_A_DIR, f"{model_name}_model.pkl"))
    
    combined = f"{clean(article)} [SEP] {clean(question)} [SEP] {clean(option_text)}"
    x_ohe = vec.transform([combined])
    
    art_vec = normalize(vec.transform([clean(article)]), axis=1)
    opt_vec = normalize(vec.transform([clean(option_text)]), axis=1)
    cos_sim = np.array(art_vec.multiply(opt_vec).sum(axis=1)).flatten().reshape(-1, 1)
    
    x_feat = sp.hstack([x_ohe, sp.csr_matrix(cos_sim)])
    x_scaled = scaler.transform(x_feat)
    
    pred = model.predict(x_scaled)[0]
    prob = model.predict_proba(x_scaled)[0][1] if hasattr(model, "predict_proba") else (0.99 if pred == 1 else 0.01)
    latency = time.time() - start_time
    return pred, prob, latency

def verify_all_options(article, question, options_dict, model_name="lr"):
    results = {}
    total_latency = 0
    for letter, text in options_dict.items():
        pred, prob, lat = verify_answer(article, question, text, model_name)
        results[letter] = {"text": text, "pred": pred, "prob": prob}
        total_latency += lat
    return results, total_latency

def generate_distractors(article, question, correct_answer, model_name="lr"):
    start_time = time.time()
    vec = _load(os.path.join(PROCESSED_DIR, "one_hot_vectorizer.pkl"))
    model = _load(os.path.join(MODEL_B_DIR, f"distractor_{model_name}.pkl"))
    scaler = _load(os.path.join(MODEL_B_DIR, "distractor_scaler.pkl")) if model_name == "lr" else None
    
    tokens = tokenize(article)
    cands = [w for w, _ in Counter([t for t in tokens if t not in STOPWORDS and len(t)>2]).most_common(20) if w != clean(correct_answer)]
    if len(cands) < 3: return ["Option 1", "Option 2", "Option 3"], time.time() - start_time
    
    cand_vecs, ans_vec = normalize(vec.transform(cands), axis=1), normalize(vec.transform([clean(correct_answer)]), axis=1)
    cos_sims = np.array(cand_vecs.multiply(ans_vec).sum(axis=1)).flatten()
    
    feats = []
    for i, cand in enumerate(cands):
        feats.append([cos_sims[i], len(set(cand) & set(correct_answer)) / max(len(set(cand) | set(correct_answer)), 1), tokens.count(cand)/max(len(tokens),1), len(cand.split()), 0.0])
    
    X = np.array(feats, dtype=np.float32)
    if scaler: X = scaler.transform(X)
    
    probs = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)
    top_idx = np.argsort(probs)[::-1][:3]
    
    latency = time.time() - start_time
    return [cands[i] for i in top_idx], latency

def gen_q(art):
    """Generates and ranks a question from the article (Model A Generation Task)"""
    start = time.time()
    vec = _load(os.path.join(PROCESSED_DIR, "one_hot_vectorizer.pkl"))
    nb = _load(os.path.join(MODEL_A_DIR, "nb_model.pkl"))
    
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', str(art)) if len(s.split()) > 5]
    if not sents: return "What is the main idea of the passage?", time.time() - start
    
    # Apply Wh-templates to extract candidates
    cands = []
    for s in sents[:5]:
        cands.append(f"What does the author imply when stating: {s}?")
        cands.append(f"Why is it significant that {s.lower()}?")
    
    # Rank candidates using the NB Question Type model
    x_ohe = vec.transform(cands)
    probs = nb.predict_proba(x_ohe)[:, 0] if hasattr(nb, "predict_proba") else [1]*len(cands)
    best = np.argmax(probs)
    
    lat = time.time() - start
    return cands[best], lat

def generate_hints(article, question, model_name="clf"):
    start_time = time.time()
    model = _load(os.path.join(MODEL_B_DIR, f"hint_{model_name}.pkl"))
    scaler = _load(os.path.join(MODEL_B_DIR, "hint_scaler.pkl"))
    
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', article) if len(s.split()) > 3]
    if not sents: return ["Read carefully."], time.time() - start_time
    
    feats = []
    for pos, sent in enumerate(sents):
        s_tok = set(tokenize(sent))
        feats.append([jaccard(s_tok, set(t for t in tokenize(question) if t not in STOPWORDS)), pos / max(len(sents) - 1, 1), min(len(sent.split()) / 50.0, 1.0), jaccard(s_tok, set(t for t in tokenize(article) if t not in STOPWORDS))])
    
    X = scaler.transform(np.array(feats, dtype=np.float32))
    probs = model.predict_proba(X)[:, 1]
    ranked_sents = [sents[i] for i in np.argsort(probs)[::-1]]
    
    hints = [
        f"General Clue: Look at the {'beginning' if np.argmax(probs) < len(sents)/2 else 'latter half'} of the passage.",
        f"Specific Clue: Pay attention to the sentence talking about: '{' '.join(ranked_sents[0].split()[:5])}...'",
        f"Near-Explicit: '{ranked_sents[0]}'"
    ]
    latency = time.time() - start_time
    return hints, latency
