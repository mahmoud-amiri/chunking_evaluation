import os
import json
import csv

# Paths
chunked_texts_folder = "./chunked_texts"
qa_pairs_file = "./chunking_evaluation/evaluation_framework/general_evaluation_data/questions_df.csv"
output_folder = "./../FSUChemRxivQest"
os.makedirs(output_folder, exist_ok=True)

# Initialize MTEB dataset dictionaries
queries = []   # Queries.jsonl (list of dicts)
corpus = []    # Corpus.jsonl (list of dicts)
qrels = []     # Qrels.jsonl (list of dicts)

# 🔹 **Step 1: Build Corpus ID to Document Name Mapping**
corpus_id_mapping = {}  # { "0": "0.txt", "1": "1.txt" }
doc_chunks = {}  # Store all chunked texts

for filename in os.listdir(chunked_texts_folder):
    if filename.endswith(".json"):
        file_path = os.path.join(chunked_texts_folder, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        if chunks and "document_name" in chunks[0]:  
            doc_name = chunks[0]["document_name"]  # e.g., "0.txt"
            doc_id = os.path.splitext(doc_name)[0]  # Remove ".txt" to match corpus_id
            corpus_id_mapping[doc_id] = doc_name  # Map corpus_id to filename
            doc_chunks[doc_name] = chunks  # Store chunks

        # Store chunked text in corpus (Ensure proper JSONL structure)
        for chunk_index, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_{chunk_index}"  # Unique chunk ID (no .txt)
            corpus.append({
                "id": chunk_id,
                "title": f"{doc_id} - Chunk {chunk_index}",
                "text": chunk["chunk_text"]
            })

print("🔄 Corpus ID Mapping:", corpus_id_mapping)  # Debugging

# 🔹 **Step 2: Load QA Pairs Safely**
qa_pairs = []
with open(qa_pairs_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        question = row["question"].strip()
        try:
            references = json.loads(row["references"])  # ✅ Safe JSON parsing
        except json.JSONDecodeError:
            print(f"❌ JSON Error in references field: {row['references']}")
            continue  # Skip bad entries

        corpus_id = row["corpus_id"].strip()  # Ensure corpus_id is string

        if corpus_id in corpus_id_mapping:
            actual_doc_name = corpus_id_mapping[corpus_id]  # Get filename
            qa_pairs.append((question, references, actual_doc_name))
        else:
            print(f"⚠️ Skipping unmatched corpus_id {corpus_id}")

# 🔹 **Step 3: Process QA Pairs & Assign Relevance**
query_id = 0
for question, references, doc_name in qa_pairs:
    queries.append({"id": str(query_id), "title": f"Query {query_id}", "text": question})  # ✅ Add title

    if doc_name not in doc_chunks:
        print(f"⚠️ Warning: Document {doc_name} not found in chunked texts.")
        query_id += 1
        continue  # Skip if the document isn't chunked

    chunks = doc_chunks[doc_name]  # Retrieve corresponding chunks
    found_match = False  # Track if any chunk matches

    for ref in references:
        start_idx = ref["start_index"]
        end_idx = ref["end_index"]
        ref_content = ref["content"]

        for chunk_index, chunk in enumerate(chunks):
            chunk_text = chunk["chunk_text"]
            chunk_start = chunk["start_index"]
            chunk_end = chunk["end_index"]
            chunk_id = f"{os.path.splitext(doc_name)[0]}_{chunk_index}"  # Remove ".txt"

            # **Matching Logic**
            if (start_idx >= chunk_start and end_idx <= chunk_end) or \
               (chunk_start <= start_idx < chunk_end) or \
               (chunk_start < end_idx <= chunk_end) or \
               (ref_content.lower() in chunk_text.lower()):
                
                qrels.append({
                    "query_id": str(query_id),
                    "doc_id": chunk_id,
                    "relevance": 1
                })
                found_match = True

    if not found_match:
        print(f"⚠️ Warning: No matching chunk found for query {query_id} in {doc_name}")

    query_id += 1

# 🔹 **Step 4: Save as JSONL files (MTEB format)**
def save_jsonl(data, file_name):
    with open(os.path.join(output_folder, file_name), "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

save_jsonl(queries, "queries.jsonl")       # ✅ Fix query format
save_jsonl(corpus, "corpus.jsonl")         # ✅ Fix corpus format
save_jsonl(qrels, "qrels.jsonl")           # ✅ Fix qrels format

print("✅ MTEB dataset created successfully!")
