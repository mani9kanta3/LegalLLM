"""
Streamlit frontend for LegalLLM.

Features:
- Ask legal questions
- Toggle between base and fine-tuned model
- Side-by-side comparison view
- Evaluation metrics dashboard
"""

import streamlit as st
import requests
import json

# === Page Config ===
st.set_page_config(
    page_title="LegalLLM — Indian Legal QA",
    page_icon="⚖️",
    layout="wide",
)

# API endpoint
API_URL = "http://localhost:8000"


def check_api():
    """Check if the API is running."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.status_code == 200
    except:
        return False


# === Sidebar ===
st.sidebar.title("⚖️ LegalLLM")
st.sidebar.markdown(
    "Fine-tuned LLM for Indian Labour Law QA\n\n"
    "**Model:** Mistral-7B + QLoRA\n\n"
    "**Dataset:** 555 QA pairs from 9 Indian acts\n\n"
    "**Method:** QLoRA (4-bit quantization + LoRA adapters)"
)

page = st.sidebar.radio("Navigate", ["Ask a Question", "Compare Models", "Evaluation Metrics", "About"])

# Check API status
api_status = check_api()
if api_status:
    st.sidebar.success("API Status: Connected ✅")
else:
    st.sidebar.error("API Status: Not Connected ❌")
    st.sidebar.caption("Run: `uvicorn app.api:app --reload`")

# === Page: Ask a Question ===
# === Page: Ask a Question ===
if page == "Ask a Question":
    st.title("⚖️ Ask a Legal Question")
    st.markdown("Get answers about Indian labour and employment law.")
    
    # Model selection
    model_type = st.radio(
        "Select model:",
        ["finetuned", "base"],
        format_func=lambda x: "🎯 Fine-Tuned Model" if x == "finetuned" else "📌 Base Model",
        horizontal=True,
    )
    
    # Question input
    question = st.text_area(
        "Your question:",
        placeholder="e.g., What is the minimum number of employees required for the ESI Act to apply?",
        height=100,
    )
    
    # Example questions — these directly trigger the answer
    st.markdown("**Or try these examples:**")
    examples = [
        "What is the procedure for filing a complaint under the POSH Act?",
        "How does the Payment of Gratuity Act define continuous service?",
        "What are the conditions for laying off workers under the Industrial Disputes Act?",
    ]
    
    # Track which question to answer
    selected_question = question.strip()
    
    cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        if cols[i].button(ex[:50] + "...", key=f"ex_{i}"):
            selected_question = ex
    
    # Also trigger on the Get Answer button
    get_answer = st.button("Get Answer", type="primary", disabled=not api_status)
    
    # Show answer if we have a question (from button click OR text input + Get Answer)
    if selected_question and (selected_question != question.strip() or get_answer):
        with st.spinner("Thinking..."):
            try:
                r = requests.post(
                    f"{API_URL}/query",
                    json={"question": selected_question, "model_type": model_type},
                    timeout=30,
                )
                data = r.json()
                
                st.markdown(f"### Question")
                st.info(selected_question)
                
                st.markdown("### Answer")
                st.write(data["answer"])
                
                conf = data["confidence"]
                color = {"high": "green", "medium": "orange", "low": "red"}[conf]
                st.markdown(f"**Confidence:** :{color}[{conf.upper()}]")
                
            except Exception as e:
                st.error(f"Error: {e}")
    elif get_answer and not selected_question:
        st.warning("Please enter a question.")


# === Page: Compare Models ===
elif page == "Compare Models":
    st.title("📊 Compare Base vs Fine-Tuned Model")
    st.markdown("See how fine-tuning improved the model's answers.")
    
    question = st.text_area(
        "Your question:",
        placeholder="Enter a question to compare both models...",
        height=100,
    )
    
    if st.button("Compare", type="primary", disabled=not api_status):
        if question.strip():
            with st.spinner("Generating answers from both models..."):
                try:
                    r = requests.post(
                        f"{API_URL}/compare",
                        json={"question": question},
                        timeout=60,
                    )
                    data = r.json()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📌 Base Model")
                        st.write(data["base_answer"])
                    
                    with col2:
                        st.markdown("### 🎯 Fine-Tuned Model")
                        st.write(data["finetuned_answer"])
                    
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter a question.")


# === Page: Evaluation Metrics ===
elif page == "Evaluation Metrics":
    st.title("📈 Evaluation Results")
    st.markdown("Comparing base Mistral-7B vs QLoRA fine-tuned model on 56 test questions.")
    
    # Try to load from API, fallback to hardcoded
    metrics = {
        "rouge_l": {"base": 0.2080, "finetuned": 0.3206},
        "bertscore": {"base": 0.8545, "finetuned": 0.8908},
        "faithfulness": {"base": 1.55, "finetuned": 1.62},
        "relevance": {"base": 3.00, "finetuned": 4.29},
    }
    
    if api_status:
        try:
            r = requests.get(f"{API_URL}/metrics", timeout=5)
            data = r.json()
            metrics["rouge_l"] = data.get("rouge_l", metrics["rouge_l"])
            metrics["bertscore"] = data.get("bertscore", metrics["bertscore"])
            if "llm_judge" in data:
                metrics["faithfulness"] = {
                    "base": data["llm_judge"]["base"]["faithfulness"],
                    "finetuned": data["llm_judge"]["finetuned"]["faithfulness"],
                }
                metrics["relevance"] = {
                    "base": data["llm_judge"]["base"]["relevance"],
                    "finetuned": data["llm_judge"]["finetuned"]["relevance"],
                }
        except:
            pass
    
    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        improvement = (metrics["rouge_l"]["finetuned"] - metrics["rouge_l"]["base"]) / metrics["rouge_l"]["base"] * 100
        st.metric("ROUGE-L F1", f"{metrics['rouge_l']['finetuned']:.4f}", f"+{improvement:.1f}%")
    
    with col2:
        improvement = (metrics["bertscore"]["finetuned"] - metrics["bertscore"]["base"]) / metrics["bertscore"]["base"] * 100
        st.metric("BERTScore F1", f"{metrics['bertscore']['finetuned']:.4f}", f"+{improvement:.1f}%")
    
    with col3:
        delta = metrics["faithfulness"]["finetuned"] - metrics["faithfulness"]["base"]
        st.metric("Faithfulness", f"{metrics['faithfulness']['finetuned']:.2f}/5", f"{delta:+.2f}")
    
    with col4:
        delta = metrics["relevance"]["finetuned"] - metrics["relevance"]["base"]
        st.metric("Relevance", f"{metrics['relevance']['finetuned']:.2f}/5", f"{delta:+.2f}")
    
    st.markdown("---")
    
    # Detailed table
    st.markdown("### Detailed Comparison")
    
    import pandas as pd
    df = pd.DataFrame({
        "Metric": ["ROUGE-L F1", "BERTScore F1", "Faithfulness (0-5)", "Relevance (0-5)"],
        "Base Model": [
            metrics["rouge_l"]["base"],
            metrics["bertscore"]["base"],
            metrics["faithfulness"]["base"],
            metrics["relevance"]["base"],
        ],
        "Fine-Tuned": [
            metrics["rouge_l"]["finetuned"],
            metrics["bertscore"]["finetuned"],
            metrics["faithfulness"]["finetuned"],
            metrics["relevance"]["finetuned"],
        ],
    })
    df["Improvement"] = df["Fine-Tuned"] - df["Base Model"]
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("### Key Findings")
    st.markdown("""
    - **ROUGE-L improved by 54.1%** — fine-tuned model uses more domain-appropriate terminology
    - **Relevance jumped from 3.0 to 4.29** — model learned to give focused, on-topic legal answers
    - **Faithfulness remains a challenge** — specific legal facts (section numbers, thresholds) need RAG augmentation
    - **Recommendation:** Hybrid approach (RAG + Fine-tuning) for production deployment
    """)


# === Page: About ===
elif page == "About":
    st.title("About LegalLLM")
    
    st.markdown("""
    ### Project Overview
    
    LegalLLM is a fine-tuned language model specialized for Indian labour and employment law 
    question answering. It uses **QLoRA (Quantized Low-Rank Adaptation)** to efficiently 
    fine-tune **Mistral-7B-Instruct** on a curated dataset of 555 QA pairs from 9 Indian 
    labour law acts.
    
    ### Technical Details
    
    | Component | Detail |
    |-----------|--------|
    | Base Model | Mistral-7B-Instruct-v0.3 |
    | Method | QLoRA (4-bit NF4 + LoRA r=16) |
    | Trainable Parameters | 13.6M (0.19% of 7.2B) |
    | Adapter Size | 56 MB (vs 14GB full model) |
    | Dataset | 555 QA pairs from 9 Indian acts |
    | Training Hardware | Google Colab T4 (15GB VRAM) |
    | GPU Memory Used | 4.5 GB |
    
    ### Documents Covered
    
    1. Companies Act, 2013
    2. Equal Remuneration Act, 1976
    3. Employee State Insurance Act, 1948
    4. Industrial Disputes Act, 1947
    5. Maternity Benefit Act, 1961
    6. Minimum Wages Act, 1948
    7. Payment of Gratuity Act, 1972
    8. Payment of Wages Act, 1936
    9. Prevention of Sexual Harassment (POSH) Act, 2013
    
    ### Links
    
    - [GitHub Repository](https://github.com/mani9kanta3/LegalLLM)
    - [W&B Dashboard](https://wandb.ai/pudimanikanta3-/LegalLLM)
    """)