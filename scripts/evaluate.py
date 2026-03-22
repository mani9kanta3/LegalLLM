"""
Step 9: Evaluation Pipeline — ROUGE-L, BERTScore, LLM-as-Judge

Why three metrics?
1. ROUGE-L (lexical): Do the words overlap? Catches exact phrasing matches.
   Limitation: "The cat sat on the mat" vs "The feline rested on the rug" = low ROUGE
   even though they mean the same thing.

2. BERTScore (semantic): Do the meanings match? Uses BERT embeddings to compare.
   Catches paraphrases that ROUGE misses.
   Limitation: Can't judge factual correctness — "The ESI Act requires 5 employees"
   vs "The ESI Act requires 500 employees" might get high BERTScore.

3. LLM-as-Judge (faithfulness): Is the answer actually correct?
   We ask Gemini to score each answer on faithfulness (0-5) and relevance (0-5).
   This catches hallucinations that lexical/semantic metrics miss.

Together, these three cover:
- Surface-level similarity (ROUGE)
- Meaning similarity (BERTScore)
- Factual correctness (LLM-as-Judge)

Usage: python scripts/evaluate.py --predictions results/predictions.json
"""

import os
import json
import argparse
import time
from collections import defaultdict

# === ROUGE-L ===
from rouge_score import rouge_scorer

# === BERTScore ===
# Note: First run downloads a BERT model (~400MB)
from bert_score import score as bert_score_fn


def compute_rouge(predictions, references):
    """
    Compute ROUGE-L scores.
    
    ROUGE-L = Longest Common Subsequence based metric
    - Precision: How much of the prediction is relevant?
    - Recall: How much of the reference is captured?
    - F1: Harmonic mean of precision and recall
    
    Returns: dict with individual scores and averages
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    
    scores = []
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        scores.append({
            "precision": score["rougeL"].precision,
            "recall": score["rougeL"].recall,
            "f1": score["rougeL"].fmeasure,
        })
    
    # Calculate averages
    avg = {
        "precision": sum(s["precision"] for s in scores) / len(scores),
        "recall": sum(s["recall"] for s in scores) / len(scores),
        "f1": sum(s["f1"] for s in scores) / len(scores),
    }
    
    return {"individual": scores, "average": avg}


def compute_bertscore(predictions, references):
    """
    Compute BERTScore for semantic similarity.
    
    How it works:
    - Encodes both prediction and reference using BERT
    - Computes cosine similarity between token embeddings
    - Matches tokens greedily for best alignment
    - Returns precision, recall, F1
    
    Higher = more semantically similar (scale: 0 to 1, typically 0.6-0.95)
    """
    P, R, F1 = bert_score_fn(
        predictions, 
        references, 
        lang="en", 
        verbose=True,
        model_type="microsoft/deberta-xlarge-mnli"  # Best model for BERTScore
    )
    
    scores = []
    for p, r, f in zip(P.tolist(), R.tolist(), F1.tolist()):
        scores.append({
            "precision": p,
            "recall": r,
            "f1": f,
        })
    
    avg = {
        "precision": sum(s["precision"] for s in scores) / len(scores),
        "recall": sum(s["recall"] for s in scores) / len(scores),
        "f1": sum(s["f1"] for s in scores) / len(scores),
    }
    
    return {"individual": scores, "average": avg}


def compute_llm_judge(predictions, references, questions, api_key):
    """
    Use Gemini as LLM-as-Judge to score faithfulness and relevance.
    
    Why LLM-as-Judge?
    - ROUGE and BERTScore can't check if facts are correct
    - A hallucinated answer with fluent language might score high on both
    - Gemini reads the question, reference answer, and model's answer
    - Scores on faithfulness (0-5) and relevance (0-5)
    
    Scoring rubric:
    5 = Perfect, accurate, complete
    4 = Good, minor omissions
    3 = Acceptable, some inaccuracies
    2 = Poor, significant errors
    1 = Bad, mostly wrong
    0 = Completely wrong or irrelevant
    """
    from google import genai
    
    client = genai.Client(api_key=api_key)
    
    scores = []
    
    for i, (pred, ref, question) in enumerate(zip(predictions, references, questions)):
        prompt = f"""You are evaluating a legal AI assistant's answer about Indian labour law.

QUESTION: {question}

REFERENCE ANSWER (ground truth): {ref}

MODEL'S ANSWER: {pred}

Score the model's answer on two dimensions:

1. FAITHFULNESS (0-5): Is the answer factually correct compared to the reference?
   5 = Perfectly accurate
   4 = Mostly accurate, minor omissions
   3 = Partially accurate, some errors
   2 = Significant factual errors
   1 = Mostly incorrect
   0 = Completely wrong

2. RELEVANCE (0-5): Does the answer address the question asked?
   5 = Directly and completely addresses the question
   4 = Addresses the question with minor gaps
   3 = Partially addresses the question
   2 = Tangentially related
   1 = Barely related
   0 = Completely irrelevant

Respond ONLY with JSON:
{{"faithfulness": <score>, "relevance": <score>, "reasoning": "<brief explanation>"}}"""

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            response_text = response.text.strip()
            response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            result = json.loads(response_text)
            scores.append({
                "faithfulness": result["faithfulness"],
                "relevance": result["relevance"],
                "reasoning": result.get("reasoning", ""),
            })
            
            print(f"  [{i+1}/{len(predictions)}] Faith={result['faithfulness']}/5, Rel={result['relevance']}/5")
            
        except Exception as e:
            print(f"  [{i+1}/{len(predictions)}] ERROR: {e}")
            scores.append({"faithfulness": -1, "relevance": -1, "reasoning": f"Error: {e}"})
        
        # Rate limiting
        time.sleep(3)
    
    # Calculate averages (excluding errors)
    valid_scores = [s for s in scores if s["faithfulness"] >= 0]
    avg = {
        "faithfulness": sum(s["faithfulness"] for s in valid_scores) / len(valid_scores) if valid_scores else 0,
        "relevance": sum(s["relevance"] for s in valid_scores) / len(valid_scores) if valid_scores else 0,
        "num_evaluated": len(valid_scores),
        "num_errors": len(scores) - len(valid_scores),
    }
    
    return {"individual": scores, "average": avg}


def main():
    parser = argparse.ArgumentParser(description="Evaluate model predictions")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSON file")
    parser.add_argument("--skip-bertscore", action="store_true", help="Skip BERTScore (needs GPU)")
    parser.add_argument("--skip-llm-judge", action="store_true", help="Skip LLM-as-Judge")
    args = parser.parse_args()
    
    # Load predictions
    with open(args.predictions, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    questions = [item["question"] for item in data]
    references = [item["reference_answer"] for item in data]
    base_preds = [item["base_model_answer"] for item in data]
    ft_preds = [item["finetuned_answer"] for item in data]
    
    print(f"Loaded {len(questions)} predictions\n")
    
    results = {"base_model": {}, "finetuned_model": {}}
    
    # === ROUGE-L ===
    print("=" * 70)
    print("ROUGE-L EVALUATION")
    print("=" * 70)
    
    print("\nBase model:")
    base_rouge = compute_rouge(base_preds, references)
    print(f"  ROUGE-L F1: {base_rouge['average']['f1']:.4f}")
    results["base_model"]["rouge_l"] = base_rouge["average"]
    
    print("\nFine-tuned model:")
    ft_rouge = compute_rouge(ft_preds, references)
    print(f"  ROUGE-L F1: {ft_rouge['average']['f1']:.4f}")
    results["finetuned_model"]["rouge_l"] = ft_rouge["average"]
    
    improvement = (ft_rouge['average']['f1'] - base_rouge['average']['f1']) / base_rouge['average']['f1'] * 100
    print(f"\n  Improvement: {improvement:+.1f}%")
    
    # === BERTScore ===
    if not args.skip_bertscore:
        print("\n" + "=" * 70)
        print("BERTSCORE EVALUATION")
        print("=" * 70)
        
        print("\nBase model:")
        base_bert = compute_bertscore(base_preds, references)
        print(f"  BERTScore F1: {base_bert['average']['f1']:.4f}")
        results["base_model"]["bertscore"] = base_bert["average"]
        
        print("\nFine-tuned model:")
        ft_bert = compute_bertscore(ft_preds, references)
        print(f"  BERTScore F1: {ft_bert['average']['f1']:.4f}")
        results["finetuned_model"]["bertscore"] = ft_bert["average"]
    
    # === LLM-as-Judge ===
    if not args.skip_llm_judge:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("\nWARNING: GEMINI_API_KEY not found. Skipping LLM-as-Judge.")
        else:
            print("\n" + "=" * 70)
            print("LLM-AS-JUDGE EVALUATION (Gemini)")
            print("=" * 70)
            
            print("\nBase model:")
            base_judge = compute_llm_judge(base_preds, references, questions, api_key)
            print(f"  Faithfulness: {base_judge['average']['faithfulness']:.2f}/5")
            print(f"  Relevance:    {base_judge['average']['relevance']:.2f}/5")
            results["base_model"]["llm_judge"] = base_judge["average"]
            
            print("\nFine-tuned model:")
            ft_judge = compute_llm_judge(ft_preds, references, questions, api_key)
            print(f"  Faithfulness: {ft_judge['average']['faithfulness']:.2f}/5")
            print(f"  Relevance:    {ft_judge['average']['relevance']:.2f}/5")
            results["finetuned_model"]["llm_judge"] = ft_judge["average"]
    
    # === Save Results ===
    output_path = os.path.join("results", "eval_results.json")
    os.makedirs("results", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    # === Print Final Comparison Table ===
    print("\n" + "=" * 70)
    print("FINAL COMPARISON: Base Model vs Fine-Tuned Model")
    print("=" * 70)
    print(f"\n{'Metric':<25} {'Base Model':>12} {'Fine-Tuned':>12} {'Change':>10}")
    print("-" * 60)
    
    if "rouge_l" in results["base_model"]:
        b = results["base_model"]["rouge_l"]["f1"]
        f = results["finetuned_model"]["rouge_l"]["f1"]
        print(f"{'ROUGE-L F1':<25} {b:>12.4f} {f:>12.4f} {f-b:>+10.4f}")
    
    if "bertscore" in results["base_model"]:
        b = results["base_model"]["bertscore"]["f1"]
        f = results["finetuned_model"]["bertscore"]["f1"]
        print(f"{'BERTScore F1':<25} {b:>12.4f} {f:>12.4f} {f-b:>+10.4f}")
    
    if "llm_judge" in results["base_model"]:
        b = results["base_model"]["llm_judge"]["faithfulness"]
        f = results["finetuned_model"]["llm_judge"]["faithfulness"]
        print(f"{'Faithfulness (0-5)':<25} {b:>12.2f} {f:>12.2f} {f-b:>+10.2f}")
        
        b = results["base_model"]["llm_judge"]["relevance"]
        f = results["finetuned_model"]["llm_judge"]["relevance"]
        print(f"{'Relevance (0-5)':<25} {b:>12.2f} {f:>12.2f} {f-b:>+10.2f}")
    
    print("-" * 60)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()