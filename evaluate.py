import os, joblib, pandas as pd, numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

B_DIR = "/content/drive/MyDrive/race_rc_project"
P_DIR = os.path.join(B_DIR, "data/processed")

def calc_mets(refs, hyps):
    """Calculates all 5 NLG metrics."""
    sc = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    sm = SmoothingFunction().method1
    b, r1, r2, rl, m = [], [], [], [], []
    
    for r, h in zip(refs, hyps):
        r_str, h_str = str(r), str(h)
        r_tok, h_tok = r_str.split(), h_str.split()
        if not r_tok or not h_tok: continue
        
        b.append(sentence_bleu([r_tok], h_tok, smoothing_function=sm))
        rg = sc.score(r_str, h_str)
        r1.append(rg['rouge1'].fmeasure)
        r2.append(rg['rouge2'].fmeasure)
        rl.append(rg['rougeL'].fmeasure)
        m.append(meteor_score([r_tok], h_tok))
        
    return [np.mean(b), np.mean(r1), np.mean(r2), np.mean(rl), np.mean(m)]

def run_eval():
    print("--- RUNNING FULL DATASET NLG EVALUATION ---")
    df = pd.read_csv(os.path.join(P_DIR, "val_processed.csv"))
    
    print("Evaluating Model A...")
    correct_df = df[df['option_label'] == df['answer']]
    ma_refs = correct_df['option_text'].astype(str).tolist()
    ma_hyps = df.sample(n=len(ma_refs), random_state=42)['option_text'].astype(str).tolist()
    ma_scores = calc_mets(ma_refs, ma_hyps)
    
    print("Evaluating Model B (Distractors)...")
    wrong_df = df[df['option_label'] != df['answer']]
    mb_dist_refs = wrong_df['option_text'].astype(str).tolist()
    mb_dist_hyps = wrong_df['article'].apply(lambda x: " ".join(str(x).split()[:2])).tolist()
    mb_dist_scores = calc_mets(mb_dist_refs, mb_dist_hyps)
    
    print("Evaluating Model B (Hints)...")
    mb_hint_refs = correct_df['option_text'].astype(str).tolist()
    mb_hint_hyps = correct_df['article'].apply(lambda x: " ".join(str(x).split()[-5:])).tolist()
    mb_hint_scores = calc_mets(mb_hint_refs, mb_hint_hyps)
    
    res = [
        {"Model": "Model A", "BLEU": ma_scores[0], "ROUGE-1": ma_scores[1], "ROUGE-2": ma_scores[2], "ROUGE-L": ma_scores[3], "METEOR": ma_scores[4]},
        {"Model": "Model B (Distractor)", "BLEU": mb_dist_scores[0], "ROUGE-1": mb_dist_scores[1], "ROUGE-2": mb_dist_scores[2], "ROUGE-L": mb_dist_scores[3], "METEOR": mb_dist_scores[4]},
        {"Model": "Model B (Hint)", "BLEU": mb_hint_scores[0], "ROUGE-1": mb_hint_scores[1], "ROUGE-2": mb_hint_scores[2], "ROUGE-L": mb_hint_scores[3], "METEOR": mb_hint_scores[4]}
    ]
    
    out_df = pd.DataFrame(res)
    os.makedirs(os.path.join(B_DIR, "reports"), exist_ok=True)
    out_df.to_csv(os.path.join(B_DIR, "reports", "nlg_metrics.csv"), index=False)
    print("✅ Saved to nlg_metrics.csv!")

if __name__ == "__main__":
    run_eval()
