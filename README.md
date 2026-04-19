# 🧠 Text-to-SQL with Prompt Tuning (PEFT + LoRA + Gemma)

Convert natural language into SQL queries using parameter-efficient fine-tuning (PEFT) with LoRA on top of a lightweight LLM.

This project demonstrates how to:

* Fine-tune a model using **instruction-style SQL datasets**
* Use **QLoRA (4-bit quantization)** for efficient training
* Perform **inference with trained adapters**
* Build a scalable **Text → SQL pipeline**

---

## 🚀 Overview

Traditional SQL generation systems rely on rules or templates. This project uses **Generative AI** to translate human language into SQL queries using a fine-tuned LLM.

### 🔁 Workflow

```
Raw Schema + Question → Prompt Formatting → Fine-Tuned LLM → SQL Query
```

---

## 📂 Project Structure

```
.
├── model_peft_training.py     # Training script (LoRA + QLoRA)
├── model_peft_inference.py    # Inference script (SQL generation)
├── sql_tuning_data.py         # Dataset generation script
├── sql_tuning_data.jsonl      # Final training dataset
├── final_sql_adapter/         # Saved LoRA adapter (output)
└── README.md
```

---

## 🧩 Dataset Preparation

Dataset is generated using structured schema + question + SQL answer format.

* Script: 
* Output: `sql_tuning_data.jsonl`

### Example Training Entry

```json
{
  "messages": [
    {"role": "system", "content": "You are a SQL expert."},
    {"role": "user", "content": "Schema: student(...)\nQuestion: List students older than 18"},
    {"role": "assistant", "content": "SELECT * FROM student WHERE age > 18;"}
  ]
}
```

---

## 🏋️ Model Training (PEFT + LoRA)

Training script: 

### Key Features

* ✅ Base Model: `google/gemma-3-1b-it`
* ✅ 4-bit Quantization (QLoRA)
* ✅ LoRA applied to all linear layers
* ✅ Efficient training with TRL `SFTTrainer`

### ⚙️ Training Config Highlights

```python
r=16
lora_alpha=32
lora_dropout=0.05
learning_rate=2e-4
batch_size=4
epochs=3
```

### ▶️ Run Training

```bash
uv venv .venv
uv sync
python model_peft_training.py
```

### 📌 Output

```
./final_sql_adapter/
```

This contains only **delta weights**, not the full model.

---

## 🔍 Inference (Generate SQL)

Inference script: 

### Steps

1. Load base model
2. Load LoRA adapter
3. Provide prompt (schema + question)
4. Generate SQL

### ▶️ Run Inference

```bash
python model_peft_inference.py
```

### 🧪 Example

**Input:**

```
Table: employees, Columns: id, name, salary  
Query: Show me the names of employees earning over 50000
```

**Output:**

```sql
SELECT name FROM employees WHERE salary > 50000;
```

---

## ⚡ Key Concepts

### 🔹 PEFT (Parameter Efficient Fine-Tuning)

Only trains a small subset of parameters → faster + cheaper

### 🔹 LoRA (Low-Rank Adaptation)

Injects trainable rank decomposition into transformer layers

### 🔹 QLoRA

Combines:

* 4-bit quantization
* LoRA adapters
  → Enables training on consumer GPUs

---

## 🛠️ Tech Stack

* 🤗 Transformers
* 🤗 Datasets
* TRL (SFTTrainer)
* PEFT (LoRA)
* PyTorch
* BitsAndBytes (4-bit quantization)

---

## 📊 Improvements You Can Add

* ✅ Schema linking (table relationships awareness)
* ✅ Multi-table join reasoning
* ✅ Execution validation (SQL correctness check)
* ✅ RAG for schema retrieval
* ✅ Fine-tuning with real enterprise datasets

---

## ⚠️ Known Limitations

* Model may hallucinate columns/tables if schema is unclear
* Complex nested queries may require more training data
* Performance depends heavily on prompt format consistency

---

## 💡 Tips

* Keep **prompt format consistent** between training and inference
* Use **low temperature (0.1)** for structured outputs like SQL
* Add more **edge-case queries** to improve robustness

---

## 📌 Future Enhancements

* Deploy as API (FastAPI / Flask)
* Integrate with Databricks SQL warehouse
* Add UI for conversational querying
* Implement evaluation metrics (execution accuracy)

---

## 🤝 Contributing

Feel free to fork and enhance:

* Better datasets
* Optimized prompts
* Larger models (Gemma 7B / LLaMA)

---

## 📜 License

MIT License

---
