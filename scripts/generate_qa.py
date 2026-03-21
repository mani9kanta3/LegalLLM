"""
Step 3: Generate QA pairs from extracted legal text using Gemini.

Usage: python scripts/generate_qa.py
"""

import os
import json
import time
from dotenv import load_dotenv
from google import genai

# Load API keys from .env file
load_dotenv()

# Configure Gemini client (new SDK)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Model to use — gemini-2.0-flash-lite is free and fast
MODEL_NAME = "gemini-2.5-flash"


# === QA DISTRIBUTION PLAN ===
QA_TARGETS = {
    "companies_act_2013": 100,
    "esi_act_1948": 80,
    "industrial_disputes_act_1947": 80,
    "minimum_wages_act_1948": 70,
    "payment_of_wages_act_1936": 60,
    "posh_act_2013": 55,
    "payment_of_gratuity_act_1972": 45,
    "equal_remuneration_act_1976": 35,
    "maternity_benefit_act_1961": 30,
}


def split_text_into_chunks(text, chunk_size=3000):
    """
    Split document text into chunks of approximately chunk_size words.
    Smaller chunks = more focused, higher quality QA pairs.
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.split()) > 100:  # Skip tiny leftover chunks
            chunks.append(chunk)

    return chunks


def generate_qa_from_chunk(chunk, doc_name, chunk_num, num_pairs):
    """
    Send a text chunk to Gemini and get back QA pairs.
    """

    prompt = f"""You are a legal expert creating a question-answer dataset from Indian labour law documents.

Given the following text from "{doc_name.replace('_', ' ').title()}", generate exactly {num_pairs} question-answer pairs.

RULES:
1. Generate three types of questions:
   - FACTUAL: Questions about specific facts, definitions, numbers, thresholds
     Example: "What is the minimum number of employees required for ESI Act applicability?"
   - PROCEDURAL: Questions about processes, steps, how things work
     Example: "What is the procedure for filing a complaint under the Industrial Disputes Act?"
   - COMPARATIVE: Questions comparing provisions, sections, or concepts within the act
     Example: "How does the definition of 'wages' differ between Section 2 and Schedule I?"

2. Answers must be 2-5 sentences long. Not one word, not an essay.
3. Do NOT copy-paste from the text. Write answers in clear, natural language.
4. Include the relevant section/chapter reference in each answer.
5. Questions should be useful for HR professionals, employees, and small business owners.
6. Mix the difficulty — some simple, some requiring understanding of multiple sections.

OUTPUT FORMAT (strict JSON array):
[
  {{
    "question": "Your question here?",
    "answer": "Your 2-5 sentence answer here. Reference Section X of the Act.",
    "question_type": "factual|procedural|comparative",
    "section_reference": "Section X / Chapter Y / Schedule Z"
  }}
]

TEXT FROM {doc_name.replace('_', ' ').upper()}:
---
{chunk}
---

Generate exactly {num_pairs} QA pairs as a JSON array. Output ONLY the JSON array, nothing else."""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        response_text = response.text.strip()

        # Clean up — Gemini sometimes wraps JSON in markdown
        response_text = response_text.replace("```json", "").replace("```", "").strip()

        # Parse JSON
        qa_pairs = json.loads(response_text)

        # Add metadata to each pair
        for pair in qa_pairs:
            pair["source_document"] = doc_name
            pair["chunk_number"] = chunk_num

        return qa_pairs

    except json.JSONDecodeError as e:
        print(f"    WARNING: Failed to parse JSON (chunk {chunk_num}): {e}")
        return []

    except Exception as e:
        print(f"    ERROR: API call failed (chunk {chunk_num}): {e}")
        return []


def process_document(doc_name, target_pairs):
    """
    Process one document: split into chunks, generate QA pairs from each.
    """
    txt_path = os.path.join("data", "processed", f"{doc_name}.txt")

    if not os.path.exists(txt_path):
        print(f"  ERROR: {txt_path} not found. Run extract_text.py first.")
        return []

    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = split_text_into_chunks(text)
    print(f"  Split into {len(chunks)} chunks")

    # Distribute QA pairs across chunks
    pairs_per_chunk = target_pairs // len(chunks)
    remainder = target_pairs % len(chunks)

    all_pairs = []

    for i, chunk in enumerate(chunks):
        num_pairs = pairs_per_chunk + (1 if i < remainder else 0)

        if num_pairs == 0:
            continue

        print(f"  Chunk {i+1}/{len(chunks)}: Generating {num_pairs} QA pairs...")

        pairs = generate_qa_from_chunk(chunk, doc_name, i + 1, num_pairs)
        all_pairs.extend(pairs)

        print(f"    Got {len(pairs)} pairs")

        # Rate limiting — 5 seconds between calls to stay within free tier
        time.sleep(5)

    return all_pairs


def main():
    output_dir = os.path.join("data", "qa_pairs")
    os.makedirs(output_dir, exist_ok=True)

    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not found.")
        print("Make sure your .env file has: GEMINI_API_KEY=your_key_here")
        return

    print("=" * 70)
    print("QA PAIR GENERATION — Indian Labour Law Documents")
    print("=" * 70)

    total_generated = 0
    all_qa_pairs = []

    for doc_name, target in QA_TARGETS.items():
        print(f"\n{'─' * 70}")
        print(f"Document: {doc_name}")
        print(f"Target: {target} QA pairs")
        print(f"{'─' * 70}")

        # Check if this document was already generated (resume support)
        doc_output_path = os.path.join(output_dir, f"{doc_name}_qa.json")
        if os.path.exists(doc_output_path):
            with open(doc_output_path, "r", encoding="utf-8") as f:
                existing_pairs = json.load(f)
            if len(existing_pairs) >= target * 0.8:  # If we got at least 80% of target
                print(f"  SKIPPING — already have {len(existing_pairs)} pairs (file exists)")
                all_qa_pairs.extend(existing_pairs)
                total_generated += len(existing_pairs)
                continue

        pairs = process_document(doc_name, target)

        # Save per-document file
        with open(doc_output_path, "w", encoding="utf-8") as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)

        print(f"  SAVED: {len(pairs)} pairs → {doc_output_path}")
        total_generated += len(pairs)
        all_qa_pairs.extend(pairs)

    # Save combined file
    combined_path = os.path.join(output_dir, "all_qa_pairs.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_qa_pairs, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'=' * 70}")
    print("GENERATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total QA pairs generated: {total_generated}")
    print(f"Combined file: {combined_path}")

    # Count by question type
    type_counts = {}
    for pair in all_qa_pairs:
        qt = pair.get("question_type", "unknown")
        type_counts[qt] = type_counts.get(qt, 0) + 1

    print(f"\nBy question type:")
    for qt, count in sorted(type_counts.items()):
        print(f"  {qt}: {count}")

    # Count by document
    print(f"\nBy document:")
    doc_counts = {}
    for pair in all_qa_pairs:
        doc = pair.get("source_document", "unknown")
        doc_counts[doc] = doc_counts.get(doc, 0) + 1

    for doc, count in sorted(doc_counts.items()):
        target = QA_TARGETS.get(doc, "?")
        print(f"  {doc}: {count}/{target}")


if __name__ == "__main__":
    main()