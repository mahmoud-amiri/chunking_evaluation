# **Impact of Chunk Size Reduction in Fixed Token Chunking for Information Retrieval**  

## **Abstract**  
In retrieval-augmented generation (RAG) and search applications, **chunk size and overlap** significantly affect retrieval accuracy. This study systematically analyzes how reducing **Fixed Token Chunking sizes** from **512 → 256 → 128 → 64 tokens** impacts retrieval performance. Our findings show:  
- **IOU (chunk alignment) improves by 587%** (from 0.0134 to 0.0921).  
- **Precision increases by 600%** (from 0.0134 to 0.0938).  
- **Recall remains stable** (~64–70%), confirming **smaller chunks preserve relevant content**.  
These results demonstrate that **smaller chunks significantly enhance retrieval precision** while maintaining recall, making them ideal for applications requiring **highly focused retrieval**.  

---

## **1. Introduction**  
Chunking is a fundamental step in **text retrieval, question-answering (QA), and NLP systems**. **Fixed Token Chunking**, a simple method that segments text into equal-sized chunks with overlap, is widely used due to its **efficiency and predictability**. However, choosing the **optimal chunk size** is crucial for balancing **precision and recall**.  

This study investigates the impact of **reducing chunk size from 512 → 256 → 128 → 64 tokens**, analyzing how retrieval accuracy changes. Our evaluation is based on **scientific and chemistry-related texts**, using metrics such as **IOU (Intersection over Union), Recall, and Precision**.  

---

## **2. Methodology**  

### **2.1 Fixed Token Chunking Configurations**  
We test the following configurations:  
1. **512 tokens, overlap 100**  
2. **256 tokens, overlap 50**  
3. **128 tokens, overlap 25**  
4. **64 tokens, overlap 12**  

Each configuration aims to balance **retrieval precision and recall** by preserving context across chunk boundaries.  

### **2.2 Evaluation Metrics**  
To assess retrieval performance, we use:  
- **IOU Mean**: Measures how well retrieved chunks align with reference excerpts.  
- **Recall Mean**: Proportion of relevant content retrieved.  
- **Precision Omega**: Stricter precision metric that evaluates chunk relevance.  
- **Precision Mean**: Measures how much retrieved content is actually relevant.  

---

## **3. Results and Discussion**  

### **3.1 Effect of Chunk Size on Retrieval Performance**  

#### **Table 1: Performance of Fixed Token Chunking at Different Chunk Sizes**  

| **Chunk Size (tokens)** | **IOU Mean** | **Recall Mean** | **Precision Omega Mean** | **Precision Mean** |
|------------------|-----------|------------|-------------------|--------------|
| **512 (Overlap: 100)**  | 0.0134  | 0.6748  | 0.0794  | 0.0134  |
| **256 (Overlap: 50)**   | 0.0271  | 0.7007  | 0.1464  | 0.0271  |
| **128 (Overlap: 25)**   | 0.0534  | 0.7089  | 0.2461  | 0.0536  |
| **64 (Overlap: 12)**    | 0.0921  | 0.6476  | 0.3772  | 0.0938  |

---

### **3.2 Key Findings**  

1. **Smaller Chunks Drastically Improve Precision**  
   - **Precision increased by 600%** (from 0.0134 → 0.0938).  
   - This means **smaller chunks retrieve significantly less irrelevant text**, improving **retrieval quality**.  

2. **IOU Increased by 587%** (from 512 → 64 tokens)  
   - Smaller chunks provide **better alignment with ground-truth excerpts**, improving retrieval accuracy.  

3. **Recall Remains Stable (~64–70%)**  
   - Despite smaller chunk sizes, **recall does not drop significantly**, proving that relevant content is still captured.  

---

## **4. Interpretation and Trade-offs**  

### **4.1 Optimal Chunk Size Recommendation**  
📌 **64-token chunks with 12-token overlap provide the best retrieval balance**:  
- **Highest precision (+600%)**  
- **Best IOU alignment (+587%)**  
- **Slight recall drop (-4.0%)** but within acceptable limits  

However, if **recall is a priority**, chunk size **128 tokens with 25-token overlap** may be a better trade-off.  

### **4.2 When to Use Different Chunk Sizes**  
| **Scenario** | **Recommended Chunk Size** |  
|-------------|---------------------------|  
| **Precision-focused Retrieval (QA, Chatbots)** | 64 tokens, overlap 12 |  
| **Balanced Precision-Recall (General Search)** | 128 tokens, overlap 25 |  
| **Recall-focused Retrieval (Document Search)** | 256 tokens, overlap 50 |  

---

## **5. Conclusion and Future Work**  

### **5.1 Summary**  
- **Reducing chunk size from 512 → 64 tokens significantly improves precision and chunk alignment.**  
- **Fixed Token Chunking at 64 tokens achieves the best trade-off between precision and recall.**  

### **5.2 Future Directions**  
1. **Testing Overlap Adjustments**  
   - Evaluating **higher overlaps (25%)** to recover potential recall loss.  
   
2. **Applying to Other Text Domains**  
   - Testing on **medical, legal, and financial documents** to validate findings.  

3. **Comparing to Semantic Chunking**  
   - Evaluating **LLM-based segmentation** to see if semantic chunking outperforms fixed chunking.  

---

## **References**  
1. OpenAI (2024). *OpenAI API Documentation*. [Online] Available: https://platform.openai.com/docs/  
2. LangChain AI (2023). *Text Splitters for Retrieval*. [Online] Available: https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitter  

---

### **Key Takeaways for Practitioners**  
✅ **Use chunk size 64 (12 overlap) for high-precision retrieval.**  
✅ **Chunk size 512 retrieves more but contains irrelevant text.**  
✅ **For balanced recall, 128-token chunks are a good alternative.**  

Would you like to test **even smaller chunks (32 tokens) or compare with semantic chunking?** 🚀