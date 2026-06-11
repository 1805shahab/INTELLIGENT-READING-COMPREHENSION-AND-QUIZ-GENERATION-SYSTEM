import os, numpy as np, pandas as pd, scipy.sparse as sp, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

BASE = "/content/drive/MyDrive/race_rc_project"
PROCESSED_DIR = os.path.join(BASE, "data/processed")
MODEL_DIR = os.path.join(BASE, "models/model_a/traditional")
TRAIN_SAMPLE = 30000 
VAL_SAMPLE = 5000

def generate_pseudo_labels(df):
    """Generates rule-based labels for NB (Question Type) and RF (Difficulty)"""
    q_types = []
    for q in df['question'].astype(str).str.lower():
        if 'who' in q: q_types.append(0)
        elif 'where' in q: q_types.append(1)
        elif 'when' in q: q_types.append(2)
        elif 'why' in q: q_types.append(3)
        else: q_types.append(4) 
    
    difficulties = [1 if len(q.split()) > 10 else 0 for q in df['question'].astype(str)]
    return np.array(q_types), np.array(difficulties)

def run_model_a_compliant():
    print("Loading datasets and features...")
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train_processed.csv")).head(TRAIN_SAMPLE)
    val_df = pd.read_csv(os.path.join(PROCESSED_DIR, "val_processed.csv")).head(VAL_SAMPLE)
    
    X_tr_ohe = sp.load_npz(os.path.join(PROCESSED_DIR, "X_train_ohe.npz"))[:TRAIN_SAMPLE]
    X_val_ohe = sp.load_npz(os.path.join(PROCESSED_DIR, "X_val_ohe.npz"))[:VAL_SAMPLE]
    cos_tr = np.load(os.path.join(PROCESSED_DIR, "cos_train.npy"))[:TRAIN_SAMPLE]
    cos_val = np.load(os.path.join(PROCESSED_DIR, "cos_val.npy"))[:VAL_SAMPLE]
    
    X_tr_verify = sp.hstack([X_tr_ohe, sp.csr_matrix(cos_tr)])
    X_val_verify = sp.hstack([X_val_ohe, sp.csr_matrix(cos_val)])
    y_tr_verify = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))[:TRAIN_SAMPLE]
    y_val_verify = np.load(os.path.join(PROCESSED_DIR, "y_val.npy"))[:VAL_SAMPLE]

    scaler = MaxAbsScaler()
    X_tr_s = scaler.fit_transform(X_tr_verify)
    X_val_s = scaler.transform(X_val_verify)

    print("\n--- Answer Verification Models ---")
    lr = LogisticRegression(max_iter=500, random_state=42).fit(X_tr_s, y_tr_verify)
    print(f"Logistic Regression Accuracy: {accuracy_score(y_val_verify, lr.predict(X_val_s)):.4f}")
    
    svm = LinearSVC(random_state=42).fit(X_tr_s, y_tr_verify)
    print(f"SVM Accuracy: {accuracy_score(y_val_verify, svm.predict(X_val_s)):.4f}")

    print("\n--- Ensemble Model (5 Marks) ---")
    ensemble = VotingClassifier(estimators=[('lr', lr), ('svm', svm)], voting='hard')
    ensemble.fit(X_tr_s, y_tr_verify)
    print(f"Ensemble (LR + SVM) Accuracy: {accuracy_score(y_val_verify, ensemble.predict(X_val_s)):.4f}")

    print("\n--- Realigned Task: Naive Bayes (Question Type) ---")
    y_tr_qt, y_val_qt = generate_pseudo_labels(train_df)[0], generate_pseudo_labels(val_df)[0]
    nb = MultinomialNB().fit(X_tr_ohe, y_tr_qt) 
    print(f"Naive Bayes (Question Type) Accuracy: {accuracy_score(y_val_qt, nb.predict(X_val_ohe)):.4f}")

    print("\n--- Realigned Task: Random Forest (Difficulty) ---")
    lex_tr = np.load(os.path.join(PROCESSED_DIR, "lex_train.npy"))[:TRAIN_SAMPLE]
    lex_val = np.load(os.path.join(PROCESSED_DIR, "lex_val.npy"))[:VAL_SAMPLE]
    y_tr_diff, y_val_diff = generate_pseudo_labels(train_df)[1], generate_pseudo_labels(val_df)[1]
    
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42).fit(lex_tr, y_tr_diff)
    print(f"Random Forest (Difficulty Estimation) Accuracy: {accuracy_score(y_val_diff, rf.predict(lex_val)):.4f}")

    joblib.dump(lr, os.path.join(MODEL_DIR, "lr_model.pkl"))
    joblib.dump(svm, os.path.join(MODEL_DIR, "svm_model.pkl"))
    joblib.dump(ensemble, os.path.join(MODEL_DIR, "ensemble_model.pkl"))
    joblib.dump(nb, os.path.join(MODEL_DIR, "nb_model.pkl"))
    joblib.dump(rf, os.path.join(MODEL_DIR, "rf_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "model_a_scaler.pkl"))
    print("\nAll Rubric-Compliant Models Saved!")

if __name__ == "__main__":
    run_model_a_compliant()
