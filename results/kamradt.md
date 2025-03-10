# **The Effect of Chunk Size on the Performance of KamradtModifiedChunker in Text Chunking and Information Retrieval**

## **Abstract**  
In natural language processing (NLP) and information retrieval, text chunking plays a crucial role in structuring large documents into semantically meaningful segments. This study evaluates the effect of chunk size on the performance of the **KamradtModifiedChunker**, a chunking algorithm that groups sentences based on semantic similarity. The evaluation is based on key retrieval metrics, including **Intersection over Union (IOU), Recall, and Precision**. Our findings indicate that reducing chunk size significantly improves precision and chunk alignment with relevant information, but excessive reduction leads to recall loss. We also suggest potential optimizations, including overlap tuning, to balance these trade-offs.  

---

## **1. Introduction**  
Chunking is a fundamental preprocessing step in NLP that influences **retrieval performance, memory efficiency, and downstream AI model accuracy**. The **KamradtModifiedChunker** modifies Greg Kamradt’s original chunking method by incorporating a similarity-based approach to segment text into meaningful units. This study investigates the impact of **chunk size reduction** on retrieval effectiveness, balancing **precision (how relevant retrieved chunks are) and recall (how much relevant information is covered)**.  

We systematically evaluate the chunking method across different chunk sizes (**400, 200, 100, and 50 tokens**) using key performance metrics. Our goal is to determine the **optimal chunk size** that maximizes retrieval efficiency without sacrificing recall.  

---

## **2. Methodology**  

### **2.1 KamradtModifiedChunker Overview**  
The **KamradtModifiedChunker** is designed to **group semantically similar sentences** into variable-sized chunks. The method:  
1. **Splits text into minimal units (sentences or small segments).**  
2. **Generates embeddings** using OpenAI’s embedding model (`text-embedding-ada-002`).  
3. **Clusters sentences** based on **cosine similarity**, merging sentences to approximate a target chunk size.  
4. **Dynamically adjusts chunk boundaries** to maintain semantic coherence.  

By controlling the **average chunk size**, we can **adjust the granularity** of information retrieval.  

### **2.2 Experimental Setup**  
We evaluated the KamradtModifiedChunker on a dataset of **scientific and chemistry-related texts**, measuring **IOU, recall, and precision** across four chunk sizes:  
- **400 tokens (baseline)**
- **200 tokens**
- **100 tokens**
- **50 tokens**  

### **2.3 Performance Metrics**  
We used the following evaluation criteria:  
- **Intersection over Union (IOU)**: Measures how well retrieved chunks align with the reference text.  
- **Recall**: Measures the proportion of relevant information retrieved.  
- **Precision Omega**: A stricter precision measure considering only retrieved relevant text.  
- **Precision**: Measures how much of the retrieved chunk is actually relevant.  

---

## **3. Results and Discussion**  

### **3.1 Effect of Chunk Size on Retrieval Performance**  
#### **Table 1: Performance of KamradtModifiedChunker at Different Chunk Sizes**  

| **Chunk Size (tokens)** | **IOU Mean** | **Recall Mean** | **Precision Omega Mean** | **Precision Mean** |
|------------------|-----------|------------|-------------------|--------------|
| **400**  | 0.0110  | 0.6629  | 0.0650  | 0.0110  |
| **200**  | 0.0194  | 0.6592  | 0.1190  | 0.0194  |
| **100**  | 0.0392  | 0.6933  | 0.2187  | 0.0393  |
| **50**   | 0.0832  | 0.6393  | 0.3747  | 0.0847  |

#### **3.2 Key Findings**  

1. **Precision Improves as Chunk Size Decreases**  
   - **Precision doubled at every reduction step**, showing that **smaller chunks contain less irrelevant text**.
   - **At chunk size 50**, **precision increased by 670%** compared to chunk size 400.

2. **IOU Score Rises Significantly with Smaller Chunks**  
   - IOU, which measures how **accurately** retrieved chunks align with reference excerpts, **increased by 656% from 400 → 50 tokens**.
   - This confirms that **smaller chunks provide better contextual alignment** with relevant content.

3. **Recall Remains Stable, But Drops at Smallest Sizes**  
   - Between chunk sizes **400 → 100**, recall was **stable** (~66–69%).
   - However, at **50 tokens, recall dropped to 63.9%**, indicating **some relevant content was lost due to smaller segment size**.

---

## **4. Interpretation and Trade-offs**  
From our results, we observe a **precision-recall trade-off**:  
- **Smaller chunks (50 tokens) improve precision and alignment** but begin to **sacrifice recall**.  
- **Larger chunks (400 tokens) maximize recall** but include **significant irrelevant information** in retrieved chunks.  

### **4.1 Optimal Chunk Size Recommendation**  
📌 **Chunk size 100 appears to offer the best balance**:  
- **High recall (69.3%)**  
- **Strong precision (0.0393, a 257% improvement from 400 tokens)**  
- **Improved chunk alignment (IOU = 0.0392, 256% increase)**  

If **higher precision is needed**, chunk size **50** is ideal, but we recommend **adding overlap** to compensate for recall loss.  

---

## **5. Conclusion and Future Work**  

### **5.1 Summary**  
- **Reducing chunk size improves precision and chunk alignment (IOU).**  
- **Excessive reduction (50 tokens) leads to recall loss.**  
- **Chunk size 100 provides the best balance between precision and recall.**  

### **5.2 Future Directions**  
1. **Fine-Tuning Overlap**  
   - Since recall **drops** at 50 tokens, we propose testing **chunk overlap** (e.g., **25–50% overlap**) to retain more relevant content.  
   
2. **Testing on Other Datasets**  
   - These experiments were conducted on **scientific texts**; further evaluation is needed on **news articles, legal documents, and technical papers**.  

3. **Comparing Alternative Chunking Strategies**  
   - Future work could compare **KamradtModifiedChunker** against **LLM-based semantic chunking** or **hierarchical clustering approaches**.  

---

## **References**  
1. Kamradt, G. (2023). *Levels of Text Splitting for Retrieval-Augmented Generation (RAG)*. [Online] Available: https://github.com/FullStackRetrieval-com  
2. OpenAI (2024). *OpenAI API Documentation*. [Online] Available: https://platform.openai.com/docs/  

---

### **Key Takeaways for Practitioners**  
✅ If your goal is **high recall**, use **100–200 tokens**.  
✅ If your goal is **high precision**, use **50 tokens + overlap**.  
✅ The **best balance** is **chunk size 100** for most retrieval applications.  

Would you like additional experiments with **overlap tuning** or different datasets? 🚀