"""
Step 5: Format QA pairs into instruction template and create train/val/test splits.

Why instruction format?
- LLMs learn from examples of "how to respond"
- The system prompt tells the model its role
- The user/assistant format matches how people will query it later
- SFTTrainer from HuggingFace TRL expects this format

Why stratified split?
- Random split might put ALL maternity act questions in training
  and NONE in test — making evaluation meaningless
- Stratified split ensures every document is represented in every split
- This way, test set performance reflects ALL legal topics

Usage: python scripts/prepare_dataset.py
"""

import os
import json
import random
from collections import defaultdict

# We'll save as simple JSON files that we can load in Colab
# No need for HuggingFace datasets library locally — we'll convert in Colab


# === INSTRUCTION TEMPLATE ===
# This is the format every training sample will follow.
# For Mistral-Instruct, the actual chat template uses [INST] tags,
# but SFTTrainer handles that automatically if we provide messages format.
# We'll use the "messages" format which works with any chat model.

SYSTEM_PROMPT = (
    "You are a legal expert specializing in Indian labour and employment law. "
    "Answer questions accurately based on Indian statutes and regulations. "
    "If unsure, say so."
)


def format_as_messages(qa_pair):
    """
    Convert a QA pair into the 'messages' format.
    
    Why messages format?
    - It's the universal format that works with ANY chat model
    - SFTTrainer automatically applies the model's chat template
    - No need to manually add [INST] or <|user|> tags
    
    The format is:
    [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
    """
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": qa_pair["question"]},
            {"role": "assistant", "content": qa_pair["answer"]},
        ],
        # Keep metadata for analysis (won't be fed to model)
        "metadata": {
            "source_document": qa_pair.get("source_document", "unknown"),
            "question_type": qa_pair.get("question_type", "unknown"),
            "section_reference": qa_pair.get("section_reference", ""),
        }
    }


def stratified_split(pairs, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Split QA pairs into train/val/test, stratified by source document.
    
    How stratification works:
    - Group all pairs by their source document
    - Within each group, shuffle and split 80/10/10
    - This ensures every split has questions from every document
    
    Example: if maternity_act has 30 pairs:
      train gets 24, val gets 3, test gets 3
    """
    random.seed(seed)  # Reproducibility — same split every time
    
    # Group by document
    by_document = defaultdict(list)
    for pair in pairs:
        doc = pair.get("source_document", "unknown")
        by_document[doc].append(pair)
    
    train_set = []
    val_set = []
    test_set = []
    
    for doc, doc_pairs in by_document.items():
        random.shuffle(doc_pairs)
        
        n = len(doc_pairs)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_set.extend(doc_pairs[:train_end])
        val_set.extend(doc_pairs[train_end:val_end])
        test_set.extend(doc_pairs[val_end:])
    
    # Shuffle each split (so documents are interleaved during training)
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)
    
    return train_set, val_set, test_set


def main():
    input_path = os.path.join("data", "qa_pairs", "all_qa_pairs_cleaned.json")
    output_dir = os.path.join("data", "dataset")
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(input_path):
        print("ERROR: all_qa_pairs_cleaned.json not found. Run validate_qa.py first.")
        return
    
    # Load cleaned QA pairs
    with open(input_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    
    print(f"Loaded {len(pairs)} cleaned QA pairs\n")
    
    # === 1. Format into messages ===
    print("[1] Formatting into instruction template...")
    formatted = [format_as_messages(pair) for pair in pairs]
    print(f"    Formatted {len(formatted)} samples")
    
    # Show one example so you can see what it looks like
    print(f"\n    === EXAMPLE FORMATTED SAMPLE ===")
    example = formatted[0]
    for msg in example["messages"]:
        role = msg["role"].upper()
        content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
        print(f"    [{role}]: {content}")
    print(f"    Metadata: {example['metadata']}")
    print(f"    === END EXAMPLE ===\n")
    
    # === 2. Stratified split ===
    print("[2] Creating stratified train/val/test split...")
    train, val, test = stratified_split(formatted)
    
    print(f"    Train: {len(train)} samples ({len(train)/len(formatted)*100:.1f}%)")
    print(f"    Val:   {len(val)} samples ({len(val)/len(formatted)*100:.1f}%)")
    print(f"    Test:  {len(test)} samples ({len(test)/len(formatted)*100:.1f}%)")
    
    # === 3. Verify stratification ===
    print(f"\n[3] Verifying document coverage in each split:")
    for split_name, split_data in [("Train", train), ("Val", val), ("Test", test)]:
        doc_counts = defaultdict(int)
        for item in split_data:
            doc_counts[item["metadata"]["source_document"]] += 1
        
        print(f"\n    {split_name}:")
        for doc, count in sorted(doc_counts.items()):
            print(f"      {doc}: {count}")
    
    # === 4. Save datasets ===
    print(f"\n[4] Saving datasets...")
    
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        filepath = os.path.join(output_dir, f"{split_name}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        print(f"    {split_name}.json: {len(split_data)} samples")
    
    # === 5. Save test questions separately (for evaluation later) ===
    # We need just the questions from the test set to run through base & fine-tuned models
    test_questions = []
    for item in test:
        test_questions.append({
            "question": item["messages"][1]["content"],  # User message
            "reference_answer": item["messages"][2]["content"],  # Assistant message
            "source_document": item["metadata"]["source_document"],
            "question_type": item["metadata"]["question_type"],
        })
    
    test_q_path = os.path.join(output_dir, "test_questions.json")
    with open(test_q_path, "w", encoding="utf-8") as f:
        json.dump(test_questions, f, indent=2, ensure_ascii=False)
    print(f"    test_questions.json: {len(test_questions)} questions (for evaluation)")
    
    # === Final Summary ===
    print(f"\n{'=' * 70}")
    print("DATASET PREPARATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total samples:  {len(formatted)}")
    print(f"Train:          {len(train)}")
    print(f"Validation:     {len(val)}")
    print(f"Test:           {len(test)}")
    print(f"Output dir:     {output_dir}")
    print(f"\nFiles created:")
    print(f"  {output_dir}/train.json      — for training")
    print(f"  {output_dir}/val.json        — for validation during training")
    print(f"  {output_dir}/test.json       — for final evaluation")
    print(f"  {output_dir}/test_questions.json — questions only (for inference)")


if __name__ == "__main__":
    main()