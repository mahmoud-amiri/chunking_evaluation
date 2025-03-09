import json
import os
from chunking_evaluation.chunking import (
    FixedTokenChunker,
    ClusterSemanticChunker,
    LLMSemanticChunker,
    KamradtModifiedChunker,
)
from chunking_evaluation.evaluation_framework.general_evaluation import GeneralEvaluation

# Initialize the evaluator with built-in dataset
evaluator = GeneralEvaluation(chroma_db_path="db/")



chunking_methods = {
    "fixed": FixedTokenChunker(chunk_size=512, chunk_overlap=100),
    "cluster_semantic": ClusterSemanticChunker(),
    "llm_semantic": LLMSemanticChunker(),
    "kamradt": KamradtModifiedChunker(avg_chunk_size=400),
}


# Store results
results = {}

for method_name, chunker in chunking_methods.items():
    print(f"Running evaluation for {method_name} chunking...")

    # Run evaluation
    evaluation_metrics = evaluator.run(chunker, retrieve=5)

    # Store results
    results[method_name] = {
        "iou_mean": float(evaluation_metrics["iou_mean"]),
        "iou_std": float(evaluation_metrics["iou_std"]),
        "recall_mean": float(evaluation_metrics["recall_mean"]),
        "recall_std": float(evaluation_metrics["recall_std"]),
        "precision_omega_mean": float(evaluation_metrics["precision_omega_mean"]),
        "precision_omega_std": float(evaluation_metrics["precision_omega_std"]),
        "precision_mean": float(evaluation_metrics["precision_mean"]),
        "precision_std": float(evaluation_metrics["precision_std"]),
        "corpora_scores": evaluation_metrics["corpora_scores"]  # Storing detailed corpus results
    }

# Define output directory
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# Save results in a readable JSON format
output_path = os.path.join(output_dir, "chunking_results.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {output_path}")
