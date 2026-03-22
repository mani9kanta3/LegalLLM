"""
Step 11: Inference script — Load base model + LoRA adapter and generate answers.

This script does two things:
1. Generates predictions from BOTH base and fine-tuned models on the test set
2. Saves predictions in the format that evaluate.py expects

Why separate from evaluation?
- Inference needs GPU (loading the model)
- Evaluation (ROUGE, BERTScore) can run on CPU
- Keeping them separate means we can run inference on Colab
  and evaluation locally — flexible workflow

Usage (on Colab): python scripts/inference.py --adapter_dir /path/to/adapter --output results/predictions.json
Usage (local):    python scripts/inference.py --adapter_dir adapters/run1 --output results/predictions.json
"""

import os
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm


# System prompt — must match what we used in training
SYSTEM_PROMPT = (
    "You are a legal expert specializing in Indian labour and employment law. "
    "Answer questions accurately based on Indian statutes and regulations. "
    "If unsure, say so."
)


def load_base_model(model_name):
    """
    Load the base Mistral-7B model in 4-bit quantization.
    Same config as training — this ensures consistency.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Model loaded. GPU memory: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
    return model, tokenizer


def load_finetuned_model(base_model, adapter_dir):
    """
    Load LoRA adapter on top of the base model.
    
    This is the beauty of LoRA — the base model stays in memory,
    we just add the 56MB adapter on top. No need to load a second 
    copy of the 7B model.
    """
    print(f"Loading LoRA adapter from: {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    print(f"Adapter loaded. GPU memory: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
    return model


def generate_answer(model, tokenizer, question, max_new_tokens=256):
    """
    Generate an answer for a single question.
    
    Steps:
    1. Format the question using Mistral's chat template
    2. Tokenize and send to GPU
    3. Generate tokens one by one (autoregressive)
    4. Decode only the new tokens (skip the prompt)
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    # Apply Mistral's chat template — adds [INST] tags automatically
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,    # Slight randomness for natural answers
            top_p=0.9,          # Nucleus sampling
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated part (skip the input prompt)
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return response.strip()


def run_inference(model, tokenizer, questions, model_name="model"):
    """
    Run inference on a list of questions with a progress bar.
    """
    answers = []
    print(f"\nGenerating answers with {model_name}...")

    for q in tqdm(questions, desc=model_name):
        answer = generate_answer(model, tokenizer, q)
        answers.append(answer)

    return answers


def main():
    parser = argparse.ArgumentParser(description="Run inference with base and fine-tuned models")
    parser.add_argument("--model_name", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--adapter_dir", required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--test_data", default="data/dataset/test_questions.json")
    parser.add_argument("--output", default="results/predictions.json")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples for quick testing")
    args = parser.parse_args()

    # Load test questions
    with open(args.test_data, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    if args.max_samples:
        test_data = test_data[:args.max_samples]
        print(f"Limited to {args.max_samples} samples for testing")

    questions = [item["question"] for item in test_data]
    references = [item["reference_answer"] for item in test_data]
    print(f"Loaded {len(questions)} test questions")

    # === Step 1: Load base model and get base answers ===
    base_model, tokenizer = load_base_model(args.model_name)
    base_answers = run_inference(base_model, tokenizer, questions, "Base Model")

    # === Step 2: Load adapter and get fine-tuned answers ===
    ft_model = load_finetuned_model(base_model, args.adapter_dir)
    ft_answers = run_inference(ft_model, tokenizer, questions, "Fine-Tuned Model")

    # === Step 3: Save predictions ===
    predictions = []
    for i in range(len(questions)):
        predictions.append({
            "question": questions[i],
            "reference_answer": references[i],
            "base_model_answer": base_answers[i],
            "finetuned_answer": ft_answers[i],
            "source_document": test_data[i].get("source_document", "unknown"),
            "question_type": test_data[i].get("question_type", "unknown"),
        })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"INFERENCE COMPLETE")
    print(f"{'='*70}")
    print(f"Questions answered: {len(predictions)}")
    print(f"Predictions saved to: {args.output}")
    print(f"\nNext step: python scripts/evaluate.py --predictions {args.output}")

    # Show a few examples
    print(f"\n{'='*70}")
    print("SAMPLE COMPARISONS")
    print(f"{'='*70}")
    for i in range(min(3, len(predictions))):
        p = predictions[i]
        print(f"\n{'─'*70}")
        print(f"Q: {p['question']}")
        print(f"\n📌 BASE:      {p['base_model_answer'][:150]}...")
        print(f"\n🎯 FINE-TUNED: {p['finetuned_answer'][:150]}...")


if __name__ == "__main__":
    main()