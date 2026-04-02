# 🧠 Prompt Tuning for Text-to-SQL (Schema-Aware)

This project demonstrates how to build a **Text-to-SQL system using Prompt Tuning (PEFT)** on top of an LLM.
The model is trained to generate SQL queries from natural language questions, conditioned on the **database schema**.

---

## 🚀 Key Features

* ⚡ **Parameter-Efficient Fine-Tuning (PEFT)** using Prompt Tuning
* 🧩 **Schema-aware training** (critical for Text-to-SQL tasks)
* 🗄️ Uses **Spider dataset** for supervised training
* 🧪 Clean separation of:

  * Trainer (Singleton)
  * Inference Engine
  * Config
* 🎯 Optimized for **low GPU memory usage**
* 🔁 Gradient accumulation + AMP support

---

## 📂 Project Structure

```
src/
├── config/
│   └── training_config.py
├── trainer/
│   └── prompt_tuning_trainer.py
├── inference/
│   └── inference_engine.py
└── main.py
```

---

## 🧠 How It Works

### 1. Prompt Tuning

Instead of fine-tuning the full model, we:

* Freeze base model weights
* Train only **virtual prompt embeddings**

This drastically reduces:

* GPU usage
* Training time

---

### 2. Schema-Aware Learning (Important 🔥)

The model is trained with schema injected into the prompt:

```
Schema:
customers(id, name, city)
orders(id, customer_id, amount)

Question: Show top 5 customers by order
SQL:
```

This enables:

* Better table/column grounding
* Higher SQL accuracy

---

### 3. Label Masking (Critical)

Only the SQL part is learned:
This ensures the model:
* Doesn’t memorize prompts
* Learns actual SQL generation

---

## 📊 Dataset

We use the **Spider dataset**, a benchmark for Text-to-SQL.

* Natural language questions
* SQL queries
* Multiple database schemas

### ⚠️ Important

Schema is **NOT present in train split directly**
It is loaded separately and joined via `db_id`.

---

## ⚙️ Installation

### 1. Create virtual environment

```bash
uv venv
```

### 2. Install dependencies

```bash
uv sync
```

---

## ▶️ Run Training

```bash
uv run python src/main.py
```

---

## 💡 Example Output

**Input:**

```
Show top 5 customers by order
```

**Output:**

```sql
SELECT customer_name, COUNT(*) 
FROM orders 
GROUP BY customer_name 
ORDER BY COUNT(*) DESC 
LIMIT 5;
```

---

## ⚠️ Known Limitations

* Prompt tuning struggles with:

  * Very large schemas
  * Complex multi-join queries
* Requires careful prompt formatting
* Performance depends heavily on schema representation

---

## 🛠️ Improvements (Next Steps)

* 🔄 Switch to **QLoRA** for better performance
* 📈 Add evaluation (execution accuracy)
* 🧪 Integrate with real database (MCP pipeline)
* 🧠 Schema compression / pruning
* 💾 Model checkpointing

---

## 🧠 Tech Stack

* PyTorch
* Hugging Face Transformers
* PEFT (Prompt Tuning)
* Datasets (Spider)
* AMP (Mixed Precision)

---

## 🙌 Acknowledgements

* Spider Dataset (Text-to-SQL benchmark)
* Hugging Face ecosystem
* PEFT library

---

## 📌 Summary

This project shows how to build a **lightweight, schema-aware Text-to-SQL system** using prompt tuning—making LLMs practical without expensive fine-tuning.

---

⭐ If you found this useful, consider starring the repo!
