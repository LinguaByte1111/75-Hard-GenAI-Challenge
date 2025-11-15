# 🧠 Dolly-Tuned: Instruction-to-Response Learning via LoRA

**Dolly-Tuned** is a lightweight fine-tuning experiment that adapts **Databricks Dolly-v2-3B** to follow human instructions more accurately.  
By combining **Low-Rank Adaptation (LoRA)** with the **PEFT (Parameter-Efficient Fine-Tuning)** framework, the project shows how large models can be customized on small datasets using minimal resources.

---

## 🎯 Project Overview
A subset of the **MBZUAI LaMini-Instruction** dataset was used, containing diverse instruction–response pairs.  
Each record was reformatted into a unified prompt-response structure and tokenized with Dolly’s pretrained tokenizer.  
The training set was intentionally kept small (200 samples) to demonstrate efficiency rather than scale.

---

## ⚙️ Methodology
- **Model:** Databricks Dolly-v2-3B  
- **Technique:** LoRA adapters applied to attention layers  
- **Training:** 3 epochs | batch size 1 | learning rate 1e-4 | fp16 precision  
- **Environment:** Google Colab / Kaggle T4 GPU  
- **Frameworks:** Hugging Face Transformers, PEFT, Datasets, PyTorch, Accelerate  

---

## 💡 Example Output
**Prompt:**  
> List 5 reasons why someone should learn to cook  

**Model Response (abridged):**  
1. Encourages healthier eating habits  
2. Builds independence and creativity  
3. Saves money on meals  
4. Strengthens family and social bonds  
5. Offers a practical lifelong skill  

---

## 🚀 Key Insights
- LoRA reduced trainable parameters and GPU memory by over 95 %.  
- The fine-tuned model generated coherent, context-aware answers.  
- Parameter-efficient tuning makes LLM adaptation accessible to small teams.

---

## 🧩 Future Work
- Add automatic evaluation metrics (BLEU, ROUGE)  
- Expand dataset variety  
- Deploy the model through Gradio or Hugging Face Spaces  

---

**Author:** [Your Name]  
**License:** MIT  
**Keywords:** Dolly v2 • LoRA • PEFT • Fine-Tuning • Instruction Tuning • NLP
