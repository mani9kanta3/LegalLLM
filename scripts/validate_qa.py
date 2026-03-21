"""
Step 4: Validate and clean generated QA pairs.

What this does:
1. Fixes inconsistent question_type casing
2. Removes duplicates (similar questions)
3. Checks answer length (should be 2-5 sentences)
4. Flags potential issues for manual review
5. Saves cleaned dataset + a review file

Usage: python scripts/validate_qa.py
"""

import os
import json
import re
from collections import Counter


def load_qa_pairs(filepath):
    """Load QA pairs from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def fix_question_types(pairs):
    """
    Normalize question_type to lowercase.
    Gemini sometimes returns 'FACTUAL' or 'Factual' instead of 'factual'.
    """
    valid_types = {"factual", "procedural", "comparative"}
    fixed_count = 0

    for pair in pairs:
        qt = pair.get("question_type", "unknown").strip().lower()

        if qt in valid_types:
            pair["question_type"] = qt
        else:
            # Try to guess from the question content
            q = pair.get("question", "").lower()
            if any(w in q for w in ["how does", "differ", "compare", "difference", "vs"]):
                pair["question_type"] = "comparative"
            elif any(w in q for w in ["how to", "procedure", "process", "steps", "file"]):
                pair["question_type"] = "procedural"
            else:
                pair["question_type"] = "factual"
            fixed_count += 1

    return pairs, fixed_count


def check_answer_length(pairs):
    """
    Check that answers are 2-5 sentences.
    Flag those that are too short or too long.
    """
    issues = []

    for i, pair in enumerate(pairs):
        answer = pair.get("answer", "")
        # Simple sentence count — split on period, exclamation, question mark
        sentences = [s.strip() for s in re.split(r'[.!?]+', answer) if s.strip()]
        num_sentences = len(sentences)

        if num_sentences < 2:
            issues.append({
                "index": i,
                "issue": "answer_too_short",
                "sentences": num_sentences,
                "question": pair["question"][:80],
                "document": pair.get("source_document", "unknown"),
            })
        elif num_sentences > 7:
            issues.append({
                "index": i,
                "issue": "answer_too_long",
                "sentences": num_sentences,
                "question": pair["question"][:80],
                "document": pair.get("source_document", "unknown"),
            })

    return issues


def find_duplicates(pairs, similarity_threshold=0.8):
    """
    Find near-duplicate questions.
    Uses simple word overlap ratio — not perfect, but catches obvious dupes.
    """
    duplicates = []

    def word_set(text):
        return set(text.lower().split())

    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            words_i = word_set(pairs[i]["question"])
            words_j = word_set(pairs[j]["question"])

            if not words_i or not words_j:
                continue

            # Jaccard similarity
            overlap = len(words_i & words_j) / len(words_i | words_j)

            if overlap > similarity_threshold:
                duplicates.append({
                    "index_1": i,
                    "index_2": j,
                    "similarity": round(overlap, 2),
                    "question_1": pairs[i]["question"][:80],
                    "question_2": pairs[j]["question"][:80],
                })

    return duplicates


def remove_duplicates(pairs, duplicates):
    """Remove the second item in each duplicate pair."""
    indices_to_remove = set()
    for dup in duplicates:
        indices_to_remove.add(dup["index_2"])

    cleaned = [p for i, p in enumerate(pairs) if i not in indices_to_remove]
    return cleaned, len(indices_to_remove)


def check_missing_fields(pairs):
    """Check that all required fields are present."""
    required = ["question", "answer", "question_type", "source_document"]
    issues = []

    for i, pair in enumerate(pairs):
        missing = [f for f in required if not pair.get(f)]
        if missing:
            issues.append({
                "index": i,
                "missing_fields": missing,
                "question": pair.get("question", "N/A")[:80],
            })

    return issues


def generate_review_sample(pairs, sample_size=30):
    """
    Pick a stratified sample for manual review.
    3-4 pairs per document, mixed question types.
    """
    by_doc = {}
    for i, pair in enumerate(pairs):
        doc = pair.get("source_document", "unknown")
        if doc not in by_doc:
            by_doc[doc] = []
        by_doc[doc].append(i)

    sample_indices = []
    per_doc = max(2, sample_size // len(by_doc))

    for doc, indices in by_doc.items():
        # Take evenly spaced samples from each document
        step = max(1, len(indices) // per_doc)
        selected = indices[::step][:per_doc]
        sample_indices.extend(selected)

    sample = []
    for idx in sample_indices[:sample_size]:
        pair = pairs[idx].copy()
        pair["review_index"] = idx
        pair["status"] = "PENDING"  # You'll mark as OK / FIX / REMOVE
        sample.append(pair)

    return sample


def main():
    input_path = os.path.join("data", "qa_pairs", "all_qa_pairs.json")
    output_dir = os.path.join("data", "qa_pairs")

    if not os.path.exists(input_path):
        print("ERROR: all_qa_pairs.json not found. Run generate_qa.py first.")
        return

    # Load data
    pairs = load_qa_pairs(input_path)
    print(f"Loaded {len(pairs)} QA pairs\n")

    # === 1. Fix question types ===
    pairs, fixed_count = fix_question_types(pairs)
    print(f"[1] Fixed question types: {fixed_count} pairs corrected")

    type_counts = Counter(p["question_type"] for p in pairs)
    for qt, count in sorted(type_counts.items()):
        print(f"    {qt}: {count}")

    # === 2. Check missing fields ===
    missing_issues = check_missing_fields(pairs)
    print(f"\n[2] Missing fields: {len(missing_issues)} pairs with issues")
    if missing_issues:
        for issue in missing_issues[:5]:
            print(f"    Index {issue['index']}: missing {issue['missing_fields']}")

    # === 3. Check answer lengths ===
    length_issues = check_answer_length(pairs)
    print(f"\n[3] Answer length issues: {len(length_issues)} pairs flagged")
    too_short = sum(1 for i in length_issues if i["issue"] == "answer_too_short")
    too_long = sum(1 for i in length_issues if i["issue"] == "answer_too_long")
    print(f"    Too short (<2 sentences): {too_short}")
    print(f"    Too long (>7 sentences): {too_long}")

    # === 4. Find duplicates ===
    print(f"\n[4] Checking for duplicate questions...")
    duplicates = find_duplicates(pairs)
    print(f"    Found {len(duplicates)} duplicate pairs")
    if duplicates:
        for dup in duplicates[:5]:
            print(f"    Sim={dup['similarity']}: '{dup['question_1']}...'")

    # === 5. Remove duplicates ===
    if duplicates:
        pairs, removed_count = remove_duplicates(pairs, duplicates)
        print(f"    Removed {removed_count} duplicates → {len(pairs)} pairs remaining")

    # === 6. Generate review sample ===
    review_sample = generate_review_sample(pairs)
    review_path = os.path.join(output_dir, "review_sample.json")
    with open(review_path, "w", encoding="utf-8") as f:
        json.dump(review_sample, f, indent=2, ensure_ascii=False)
    print(f"\n[5] Review sample: {len(review_sample)} pairs saved to {review_path}")
    print(f"    MANUAL STEP: Open {review_path} and verify these pairs are correct.")

    # === 7. Save cleaned dataset ===
    cleaned_path = os.path.join(output_dir, "all_qa_pairs_cleaned.json")
    with open(cleaned_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)

    # === Final Summary ===
    print(f"\n{'=' * 70}")
    print("VALIDATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"Original pairs:     {len(load_qa_pairs(input_path))}")
    print(f"After cleaning:     {len(pairs)}")
    print(f"Duplicates removed: {len(duplicates)}")
    print(f"Length issues:      {len(length_issues)} (kept but flagged)")
    print(f"Cleaned file:       {cleaned_path}")
    print(f"Review sample:      {review_path}")

    # Document coverage
    print(f"\nDocument coverage:")
    doc_counts = Counter(p["source_document"] for p in pairs)
    for doc, count in sorted(doc_counts.items()):
        print(f"  {doc}: {count}")


if __name__ == "__main__":
    main()