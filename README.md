# Intelligent Reading Comprehension & MCQ Generation System

This project presents an end-to-end framework for **automated Reading Comprehension (RC)** and **multiple-choice question (MCQ) generation** using the RACE dataset. It combines classical machine learning methods with feature engineering techniques to build a supervised answer verification system and an extractive question/hint generation pipeline.

The system demonstrates how traditional ML approaches (Logistic Regression, SVM, K-Means) can still achieve strong performance in NLP-based educational applications.

---

## 📌 Key Features

- Supervised answer verification using ensemble learning (Logistic Regression + SVM)
- Feature engineering using:
  - One-Hot Encoding (5,000 dimensions)
  - Cosine Similarity scoring
- Extractive MCQ distractor and hint generation system
- Unsupervised clustering analysis using K-Means
- Real-time interactive quiz interface using Streamlit
- Analytics dashboard for performance and session tracking

---

## 📊 Dataset

- **Dataset:** RACE (Reading Comprehension from Examinations)
- **Size:** 13,329 samples (validation set used for evaluation)
- **Structure:** Passages with multiple-choice questions (A, B, C, D)

### Observations
- Balanced class distribution (~25% per label)
- Average passage length: ~321 words
- High linguistic complexity requiring reasoning beyond keyword matching

---

## 🧠 System Architecture

The system consists of two main components:

### Model A — Answer Verification & Ranking
- Logistic Regression + Linear SVM ensemble (hard voting)
- Feature set:
  - 5,000-dim One-Hot Encoding vectors
  - Cosine similarity between passage and answer options
- Task:
  - Answer classification
  - Question ranking

**Performance:**
- Accuracy: **73.3%**
- Macro F1 Score: **0.4576**

---

### Model B — MCQ Generation & Hint System
- Extractive distractor generation using NLP heuristics
- Random Forest-based ranking of candidate distractors
- Multi-level hint generation system:
  - Level 1: Basic keyword hints
  - Level 2: Contextual sentence hints
  - Level 3: Near-answer guided hints

**Evaluation:**
- BLEU: 0.0078  
- ROUGE-L: 0.0472  

---

## 📉 Unsupervised Analysis

- K-Means clustering applied to feature space
- Silhouette Score: **0.0063**
- Cluster purity: **0.75**

### Insight:
High feature overlap between correct and incorrect answers shows limitations of purely lexical representations.

---

## 🖥️ User Interface

Built using **Streamlit**, the system includes:

- 📄 Article Input Panel  
  - Manual input or dataset sampling  
  - Auto-question generation  

- 🧪 Quiz Interface  
  - Interactive MCQ solving  
  - Real-time confidence scoring  

- 💡 Hint System  
  - Three-level extractive hints  

- 📊 Analytics Dashboard  
  - Model performance metrics  
  - Latency tracking  
  - Session statistics  

---

## ⚙️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- FastAPI
- Streamlit
- NLP (Cosine Similarity, feature extraction)

---

## 📈 Key Insights

- Cosine similarity is the most influential feature for classification
- Vocabulary overlap causes difficulty in distinguishing "trap" distractors
- Classical ML models can still perform competitively on structured NLP tasks

---

## 🚧 Limitations

- Extractive-only generation limits linguistic fluency
- Cannot handle synonym/paraphrase reasoning well
- Weak performance on semantically similar distractors

---

## 🔮 Future Work

- Replace OHE with contextual embeddings (Word2Vec / BERT)
- Improve semantic understanding using transformer-based models
- Add TF-IDF weighting for better distractor filtering
- Integrate RAG-style retrieval for better question generation

---

## 📚 References

1. Lai et al., 2017 — RACE Dataset  
2. Pedregosa et al., 2011 — Scikit-learn  
3. Papineni et al., 2002 — BLEU Metric  
4. Manning, 2008 — Information Retrieval  
5. Jurafsky & Martin, 2023 — Speech and Language Processing  

---

## 📌 Conclusion

This project demonstrates that classical machine learning techniques, when combined with strong feature engineering, can effectively solve structured NLP tasks such as reading comprehension and MCQ generation. The system achieves solid performance while providing an interactive educational tool for real-world use.

---
