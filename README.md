# ⚖️ LegalLLM — Fine-Tuned LLM for Indian Legal Question Answering

Fine-tuned **Mistral-7B-Instruct** using **QLoRA** on a self-curated dataset of **555 QA pairs** from 9 Indian labour law documents. The model answers questions about Indian employment law with improved relevance and domain specificity compared to the base model.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                       │
│                  (Streamlit Frontend App)                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                           │
│              /query    /compare    /metrics                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
┌──────────────────────┐   ┌──────────────────────┐
│   Fine-Tuned LLM     │   │     Base Model       │
│ Mistral-7B + QLoRA   │   │   Mistral-7B-Instruct│
│ (56MB LoRA Adapter)  │   │   (14GB Full Model)  │
└──────────────────────┘   └──────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│              Evaluation & Tracking Layer                      │
│     ROUGE-L  │  BERTScore  │  LLM-as-Judge  │  W&B          │
└─────────────────────────────────────────────────────────────┘
```

---

## Results

### Base Model vs Fine-Tuned Model (56 test questions)

| Metric               | Base Model | Fine-Tuned | Change     |
|----------------------|-----------|------------|------------|
| **ROUGE-L F1**       | 0.2080    | 0.3206     | **+54.1%** |
| **BERTScore F1**     | 0.8545    | 0.8908     | **+4.2%**  |
| **Faithfulness (0-5)** | 1.55    | 1.62       | +0.07      |
| **Relevance (0-5)**  | 3.00      | 4.29       | **+1.29**  |

**Key Findings:**
- ROUGE-L improved by 54.1% — the fine-tuned model uses more domain-appropriate legal terminology
- Relevance jumped from 3.0 to 4.29 out of 5 — the model gives focused, on-topic legal answers instead of generic responses
- Faithfulness remains a challenge for both models — specific legal citations (section numbers, thresholds) require additional approaches like RAG
- **Recommendation:** A hybrid RAG + fine-tuning approach would be optimal for production, where RAG provides exact section references and fine-tuning provides fluent, domain-aware language

### Training Curves

| Step | Training Loss | Validation Loss |
|------|--------------|-----------------|
| 25   | 1.6483       | 1.5898          |
| 50   | 1.3055       | **1.4734** (best)|
| 75   | 1.1330       | 1.4833          |

Best checkpoint: **Step 50** (lowest validation loss). Mild overfitting observed after step 50, which is expected with a 444-sample dataset.

📊 **W&B Dashboard:** [https://wandb.ai/pudimanikanta3-/LegalLLM](https://wandb.ai/pudimanikanta3-/LegalLLM)

---

## Dataset

### Source Documents (9 Indian Labour Law Acts)

| # | Document                         | Pages | Words   | QA Pairs |
|---|----------------------------------|-------|---------|----------|
| 1 | Companies Act, 2013              | 370   | 206,340 | 100      |
| 2 | Employee State Insurance Act, 1948| 118  | 30,379  | 80       |
| 3 | Industrial Disputes Act, 1947    | 53    | 23,804  | 80       |
| 4 | Minimum Wages Act, 1948          | 50    | 14,719  | 70       |
| 5 | Payment of Wages Act, 1936       | 19    | 10,892  | 60       |
| 6 | POSH Act, 2013                   | 14    | 6,358   | 55       |
| 7 | Payment of Gratuity Act, 1972    | 22    | 5,892   | 45       |
| 8 | Equal Remuneration Act, 1976     | 9     | 3,573   | 35       |
| 9 | Maternity Benefit Act, 1961      | 4     | 2,022   | 30       |
|   | **TOTAL**                        | **659**| **303,979** | **555** |

### Dataset Statistics

- **Total QA pairs:** 555
- **Split:** Train 444 (80%) / Validation 55 (10%) / Test 56 (10%)
- **Stratified by:** Source document (every act is represented in every split)
- **Question types:** Factual (287), Procedural (165), Comparative (103)
- **Generation method:** Gemini 2.5 Flash with manual verification
- **Answer length:** 2-5 sentences with section references

### Question Type Examples

**Factual:** "What is the minimum number of employees required for ESI Act applicability?"

**Procedural:** "What is the procedure for filing a complaint under the POSH Act?"

**Comparative:** "How does the definition of 'wages' differ between the Payment of Wages Act and the Minimum Wages Act?"

---

## Training Details

### QLoRA Configuration

| Parameter           | Value                           |
|--------------------|---------------------------------|
| Base Model         | Mistral-7B-Instruct-v0.3        |
| Total Parameters   | 7,248,023,552 (7.2B)            |
| Trainable Parameters | 13,631,488 (0.19%)            |
| Quantization       | 4-bit NF4 with double quantization |
| LoRA Rank (r)      | 16                              |
| LoRA Alpha         | 32                              |
| LoRA Dropout       | 0.05                            |
| Target Modules     | q_proj, k_proj, v_proj, o_proj  |
| Adapter Size       | 56 MB (vs 14 GB full model)     |

### Training Arguments

| Parameter                  | Value           |
|---------------------------|-----------------|
| Epochs                    | 3               |
| Batch Size (effective)    | 16 (4 × 4 accumulation) |
| Learning Rate             | 2e-4            |
| Scheduler                 | Cosine          |
| Warmup Ratio              | 0.03            |
| Precision                 | BFloat16        |
| Optimizer                 | Paged AdamW 8-bit |
| Gradient Checkpointing    | Enabled         |
| Max Sequence Length        | 2048            |

### Hardware

| Environment | GPU | VRAM | Used For |
|------------|-----|------|----------|
| Google Colab | Tesla T4 | 15 GB | Training & Inference |
| Local | GTX 1650 Ti | 4 GB | Development & Evaluation |

GPU memory during training: **4.5 GB** (out of 15 GB available on T4)

### Hyperparameter Experiments (Planned)

| Run | LoRA Rank | Learning Rate | Epochs | Status |
|-----|-----------|---------------|--------|--------|
| Run 1 | r=16 | 2e-4 | 3 | ✅ Complete |
| Run 2 | r=8 | 2e-4 | 3 | 🔄 Planned |
| Run 3 | r=32 | 2e-4 | 3 | 🔄 Planned |
| Run 4 | r=16 | 2e-4 | 5 | 🔄 Planned |

---

## Evaluation Framework

Three-layer evaluation to capture different aspects of answer quality:

### 1. ROUGE-L (Lexical Overlap)
Measures word-level overlap between generated and reference answers. Captures terminology alignment but misses paraphrases.

### 2. BERTScore (Semantic Similarity)
Uses BERT embeddings to compare meaning. Catches paraphrases that ROUGE misses but can't verify factual correctness.

### 3. LLM-as-Judge (Faithfulness & Relevance)
Gemini 2.5 Flash scores each answer on:
- **Faithfulness (0-5):** Is the answer factually correct compared to reference?
- **Relevance (0-5):** Does the answer address the question asked?

This three-layer approach ensures we measure surface similarity, semantic meaning, AND factual correctness.

---

## Project Structure

```
LegalLLM/
├── data/
│   ├── raw/                    # 9 Indian labour law PDFs
│   ├── processed/              # Extracted text + stats
│   ├── qa_pairs/               # Generated & validated QA pairs
│   └── dataset/                # Train/Val/Test splits (JSON)
├── scripts/
│   ├── extract_text.py         # PDF → text extraction
│   ├── generate_qa.py          # Gemini-based QA generation
│   ├── validate_qa.py          # Data quality checks
│   ├── prepare_dataset.py      # Format + stratified split
│   ├── inference.py            # Base vs fine-tuned comparison
│   └── evaluate.py             # ROUGE, BERTScore, LLM-Judge
├── app/
│   ├── api.py                  # FastAPI backend
│   ├── frontend.py             # Streamlit frontend
│   └── Dockerfile
├── configs/
│   └── training_config.yaml    # All hyperparameters
├── results/
│   ├── predictions.json        # Model outputs on test set
│   └── eval_results.json       # Evaluation scores
├── requirements.txt            # Development dependencies
├── requirements-deploy.txt     # Deployment dependencies
├── start.sh                    # Docker entrypoint
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA (for local inference)
- Google Colab account (for training)

### Local Development

```bash
# Clone the repository
git clone https://github.com/mani9kanta3/LegalLLM.git
cd LegalLLM

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\Activate.ps1  # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys:
#   HUGGINGFACE_TOKEN=your_token
#   WANDB_API_KEY=your_key
#   GEMINI_API_KEY=your_key
```

### Run the Application

```bash
# Terminal 1: Start FastAPI backend
uvicorn app.api:app --reload

# Terminal 2: Start Streamlit frontend
streamlit run app/frontend.py
```

Open `http://localhost:8501` in your browser.

### Run Evaluation

```bash
# Generate predictions (requires GPU)
python scripts/inference.py --adapter_dir adapters/run1 --output results/predictions.json

# Run evaluation metrics
python scripts/evaluate.py --predictions results/predictions.json
```

### Docker

```bash
docker build -t legalllm -f app/Dockerfile .
docker run -p 8000:8000 -p 8501:8501 --env-file .env legalllm
```

---

## Limitations (Honest Assessment)

1. **Factual precision is limited** — The model improved significantly on relevance but still struggles with exact section numbers and legal thresholds. Faithfulness scores (1.55 → 1.62) show both models have room for improvement on factual accuracy.

2. **Small training dataset** — 444 training samples is small for a 7B model. The model learned domain style and terminology effectively but couldn't memorize all specific legal provisions.

3. **Section references inconsistent** — The fine-tuned model doesn't always cite specific sections in its answers, despite training data containing references. This is a known limitation being addressed in upcoming training runs.

4. **Single domain** — Currently covers only 9 Indian labour law acts. Would need additional training data for other legal domains (tax, corporate, IP law).

5. **No real-time updates** — The model's knowledge is static. New amendments or judicial interpretations won't be reflected without retraining.

**Planned improvements:**
- Retrain with longer answers (4-8 sentences) that consistently include section references
- Hyperparameter experiments (r=8, r=32, different learning rates)
- Hybrid RAG + fine-tuning pipeline for production accuracy

---

## Interview Talking Points

**Why QLoRA over full fine-tuning?**
> Full fine-tuning of Mistral-7B requires ~56GB VRAM. QLoRA freezes the base model in 4-bit (3.9GB) and trains only 0.19% of parameters via LoRA adapters. Total training memory: 4.5GB on a T4 GPU.

**How did you ensure data quality?**
> Generated QA pairs using Gemini with structured prompts, then ran automated validation (deduplication, length checks, type normalization). Manually reviewed a stratified sample of 27 pairs across all 9 documents.

**Why is faithfulness low?**
> Fine-tuning excels at teaching style, terminology, and relevance — but memorizing specific legal facts requires either more training data or retrieval augmentation. This is exactly why hybrid RAG + fine-tuning is the production recommendation.

**What would you do with more resources?**
> More training data (2000+ pairs), larger LoRA rank, hybrid RAG pipeline for citation accuracy, and evaluation on real user queries from legal professionals.

---

## Technology Stack

- **Language:** Python 3.12
- **Base Model:** Mistral-7B-Instruct-v0.3
- **Fine-Tuning:** HuggingFace Transformers, PEFT, TRL, bitsandbytes
- **Evaluation:** rouge-score, BERTScore, Gemini (LLM-as-Judge)
- **Tracking:** Weights & Biases
- **Backend:** FastAPI
- **Frontend:** Streamlit
- **Deployment:** Docker
- **Hardware:** Google Colab T4 (training), GTX 1650 Ti (development)

---

## License

This project is for educational and research purposes. The Indian legal documents used are publicly available government publications.

---

## Author

**Pudi Manikanta**
- GitHub: [@mani9kanta3](https://github.com/mani9kanta3)
- W&B: [pudimanikanta3-/LegalLLM](https://wandb.ai/pudimanikanta3-/LegalLLM)