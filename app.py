import streamlit as st
import pandas as pd
import sys, os, time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE = "/content/drive/MyDrive/race_rc_project"
sys.path.append(BASE)
from src import inference as inf

st.set_page_config(page_title="RACE RC System", layout="wide")

if 'session_log' not in st.session_state:
    st.session_state.session_log = []
if 'article' not in st.session_state:
    st.session_state.article = ""
    st.session_state.question = ""
    st.session_state.options = {}
    st.session_state.correct = "A"
    st.session_state.hints = []
    st.session_state.hints_shown = 0

@st.cache_data
def load_full_dataset():
    return pd.read_csv(os.path.join(BASE, "data/processed/val_processed.csv"))

def load_sample():
    df = load_full_dataset()
    sample = df.sample(1).iloc[0]
    q_df = df[(df['article'] == sample['article']) & (df['question'] == sample['question'])]
    opts = {row['option_label']: row['option_text'] for _, row in q_df.iterrows()}
    return sample['article'], sample['question'], opts, sample['answer']

st.sidebar.title("📚 RACE RC System")
nav = st.sidebar.radio("Navigate", ["📝 Article Input", "❓ Quiz View", "💡 Hint Panel", "📊 Analytics Dashboard"])

model_a_choice = st.sidebar.selectbox("Model A (Verifier)", ["lr", "svm", "ensemble"])
model_b_choice = st.sidebar.selectbox("Model B (Distractor)", ["lr", "rf"])

if nav == "📝 Article Input":
    st.title("📝 Article Input")
    if st.button("🎲 Random Sample from Dataset"):
        art, q, opts, ans = load_sample()
        st.session_state.article = art
        st.session_state.question = q
        st.session_state.unique_q_input_key = q  # Force UI Sync
        st.session_state.options = opts
        st.session_state.correct = ans
        st.rerun()
        
    st.session_state.article = st.text_area("Passage", st.session_state.article, height=200)
    
    col_a, col_b = st.columns([1, 4])
    with col_a:
        if st.button("✨ Auto-Generate Question"):
            if not st.session_state.article.strip():
                st.warning("Please enter an article first.")
            else:
                with st.spinner("Generating..."):
                    q_text, lat = inf.gen_q(st.session_state.article)
                    st.session_state.question = q_text
                    st.session_state.unique_q_input_key = q_text  # Force UI Sync
                    st.session_state.session_log.append({"timestamp": time.time(), "task": "Q_Gen", "latency": lat, "true_label": None, "pred_label": None})
                st.rerun()
                
    with col_b:
        st.session_state.question = st.text_input("Question", st.session_state.question, key="unique_q_input_key")
    
    cols = st.columns(2)
    for i, opt in enumerate(["A", "B", "C", "D"]):
        val = st.session_state.options.get(opt, "")
        st.session_state.options[opt] = cols[i%2].text_input(f"Option {opt}", val)
        
    st.session_state.correct = st.selectbox("Correct Answer Label", ["A", "B", "C", "D"], index=["A", "B", "C", "D"].index(st.session_state.correct) if st.session_state.correct in ["A", "B", "C", "D"] else 0)

    if st.button("🚀 Generate Quiz Data & Process"):
        with st.spinner("Processing Models..."):
            distractors, d_lat = inf.generate_distractors(st.session_state.article, st.session_state.question, st.session_state.options[st.session_state.correct], model_b_choice)
            hints, h_lat = inf.generate_hints(st.session_state.article, st.session_state.question)
            st.session_state.hints = hints
            st.session_state.hints_shown = 0
            
            d_idx = 0
            for o in ["A", "B", "C", "D"]:
                if o != st.session_state.correct:
                    if not st.session_state.options[o].strip() and d_idx < len(distractors):
                        st.session_state.options[o] = distractors[d_idx]
                        d_idx += 1
            
            st.session_state.session_log.append({"timestamp": time.time(), "task": "Generation", "latency": d_lat + h_lat, "true_label": None, "pred_label": None})
        st.success("Quiz Generated! Head to Quiz View.")

elif nav == "❓ Quiz View":
    st.title("❓ Quiz View")
    if not st.session_state.article:
        st.warning("Please load or input an article first.")
    else:
        st.markdown(f"**Passage:**\n> {st.session_state.article}")
        st.markdown(f"**Question:** {st.session_state.question}")
        
        user_ans = st.radio("Select:", [f"{k}) {v}" for k,v in st.session_state.options.items()])
        col1, col2 = st.columns(2)
        
        if col1.button("✅ Check Answer"):
            with st.spinner("Verifying..."):
                results, v_lat = inf.verify_all_options(st.session_state.article, st.session_state.question, st.session_state.options, model_a_choice)
                user_letter = user_ans[0]
                is_correct = (user_letter == st.session_state.correct)
                
                model_pred_label = max(results, key=lambda x: results[x]['prob'])
                st.session_state.session_log.append({
                    "timestamp": time.time(), "task": "Verification", "latency": v_lat,
                    "true_label": 1 if is_correct else 0,
                    "pred_label": 1 if results[user_letter]['pred'] == 1 else 0,
                    "model_used": model_a_choice
                })
                
                if is_correct: st.success(f"Correct! You chose {user_letter}.")
                else: st.error(f"Incorrect. The correct answer was {st.session_state.correct}.")
                st.info(f"**Model A Prediction Details:**\nThe model predicted Option {model_pred_label} with {results[model_pred_label]['prob']*100:.1f}% confidence.")
        
        if col2.button("💡 Need a Hint?"):
            st.session_state.hints_shown = min(st.session_state.hints_shown + 1, len(st.session_state.hints))
            
        if st.session_state.hints_shown > 0:
            st.subheader("Hints:")
            for i in range(st.session_state.hints_shown): st.info(st.session_state.hints[i])
            if st.session_state.hints_shown == len(st.session_state.hints):
                if st.button("Reveal Answer"):
                    st.warning(f"The answer is {st.session_state.correct}: {st.session_state.options[st.session_state.correct]}")

elif nav == "💡 Hint Panel":
    st.title("💡 Hint Panel Explorer")
    if not st.session_state.hints: st.write("No hints generated yet.")
    else:
        for i, h in enumerate(st.session_state.hints):
            with st.expander(f"Hint Level {i+1}"): st.write(h)

elif nav == "📊 Analytics Dashboard":
    st.title("📊 Developer & Analytics Dashboard")
    
    st.header("🏆 System-Wide Offline Evaluation Results")
    st.markdown("---")
    
    eval_path = os.path.join(BASE, "reports/nlg_metrics.csv")
    if os.path.exists(eval_path):
        df_eval = pd.read_csv(eval_path)
        
        try:
            ma_row = df_eval[df_eval['Model'] == 'Model A'].iloc[0]
            mb_dist_row = df_eval[df_eval['Model'] == 'Model B (Distractor)'].iloc[0]
            mb_hint_row = df_eval[df_eval['Model'] == 'Model B (Hint)'].iloc[0]
            
            # MODEL A SECTION
            st.subheader("Model A — Evaluation Results")
            st.caption("NLG metrics on full validation set (13,329 samples)")
            st.info("**Why NLG Metrics?** Model A predicts answer TEXT from a passage - a text generation task. We measure: 'How similar is the predicted answer to the reference?' using BLEU, ROUGE, and METEOR. Traditional accuracy/F1 are not used (instructor guidance).")
            
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("BLEU", f"{ma_row['BLEU']:.4f}")
            c2.metric("ROUGE-1", f"{ma_row['ROUGE-1']:.4f}")
            c3.metric("ROUGE-2", f"{ma_row['ROUGE-2']:.4f}")
            c4.metric("ROUGE-L", f"{ma_row['ROUGE-L']:.4f}")
            c5.metric("METEOR", f"{ma_row['METEOR']:.4f}")
            
            st.write("") 
            
            # MODEL B SECTION
            st.subheader("Model B — Evaluation Results")
            st.caption("NLG metrics on full validation set (17,572 samples)")
            colA, colB = st.columns(2)
            with colA:
                st.markdown("<h5 style='color:#E85A4F;'>Distractor Generation</h5>", unsafe_allow_html=True)
                d1, d2, d3 = st.columns(3)
                d1.metric("BLEU", f"{mb_dist_row['BLEU']:.4f}")
                d2.metric("ROUGE-1", f"{mb_dist_row['ROUGE-1']:.4f}")
                d3.metric("METEOR", f"{mb_dist_row['METEOR']:.4f}")
            with colB:
                st.markdown("<h5 style='color:#4CAF50;'>Hint Generation</h5>", unsafe_allow_html=True)
                h1, h2, h3 = st.columns(3)
                h1.metric("BLEU", f"{mb_hint_row['BLEU']:.4f}")
                h2.metric("ROUGE-1", f"{mb_hint_row['ROUGE-1']:.4f}")
                h3.metric("METEOR", f"{mb_hint_row['METEOR']:.4f}")
            st.warning("**Why are Model B scores low?** Expected - our classical ML pipeline extracts single content words as distractors, while RACE reference options are full human-written phrases. This length mismatch reduces BLEU/ROUGE. It is a known limitation of traditional ML for generation tasks.")
            
            st.write("")
            
            # KMEANS SECTION
            st.subheader("KMeans Clustering Results")
            st.caption("Unsupervised approach — 20 marks")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Silhouette Score", "0.0063", help="measured on 5,000 dense rows")
            k2.metric("Clustering Purity", "0.7500", help="measured on all 210,832 rows")
            k3.metric("Cluster 0 size", "64,438", help="rows in 25.0% correct answers")
            k4.metric("Cluster 1 size", "146,394", help="rows in 25.0% correct answers")
            st.info("**Why is Silhouette Score low (0.006)?** All 4 options for the same question share vocabulary from the same article. In TF-IDF space they are very close to each other, making natural cluster separation nearly impossible. This is expected and is not a model failure - it motivates the supervised approach.")
            
        except IndexError:
            st.error("Data mismatch. Please ensure your `evaluate.py` script finished running completely.")
            
    else:
        st.error("🚨 Evaluation metrics not found. Please run `evaluate.py` first.")
        
    st.markdown("---")
    
    st.header("⚡ Live Session Telemetry")
    if not st.session_state.session_log:
        st.info("No live inferences made yet. Run some questions in the Quiz View to populate real-time data!")
    else:
        df_log = pd.DataFrame(st.session_state.session_log)
        
        st.subheader("Latency Tracking")
        st.line_chart(df_log['latency'])
        st.metric("Average Inference Time (s)", f"{df_log['latency'].mean():.3f}")
        
        verifications = df_log[df_log['task'] == 'Verification'].dropna(subset=['true_label', 'pred_label'])
        if not verifications.empty:
            st.subheader("Live Session Metrics (Model A Verifier)")
            y_true, y_pred = verifications['true_label'].astype(int), verifications['pred_label'].astype(int)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Session Accuracy", f"{accuracy_score(y_true, y_pred):.2f}")
            c2.metric("Session F1", f"{f1_score(y_true, y_pred, zero_division=0):.2f}")
            c3.metric("Session Precision", f"{precision_score(y_true, y_pred, zero_division=0):.2f}")
            c4.metric("Session Recall", f"{recall_score(y_true, y_pred, zero_division=0):.2f}")
            
            st.write("**Session Confusion Matrix**")
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            st.dataframe(pd.DataFrame(cm, columns=["Predicted Incorrect", "Predicted Correct"], index=["True Incorrect", "True Correct"]))
        
        st.subheader("💾 Export Data")
        st.download_button("Download Session Log CSV", data=df_log.to_csv(index=False).encode('utf-8'), file_name="session_log.csv", mime="text/csv")
