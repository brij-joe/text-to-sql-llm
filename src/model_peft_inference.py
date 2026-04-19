import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. Setup paths
base_model_id = "google/gemma-3-1b-it"
adapter_path = "./final_sql_adapter"  # The folder from your previous script

# 2. Load the Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# 3. Load the Base Model (ideally in 4-bit to match training conditions)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 4. Load the PEFT Adapter
# This merges the trained "delta" weights with the base model
model = PeftModel.from_pretrained(model, adapter_path)

# 5. Prepare the prompt
# Note: Ensure this matches the format used in your training dataset!
prompt = "Table: employees, Columns: id, name, salary. Query: Show me the names of employees earning over 50000."
# Automatically use the device the model is currently on
inputs = tokenizer(prompt, return_tensors="pt").to(model.device) 

# 6. Generate SQL
model.eval() # Set to evaluation mode
with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=100,
        temperature=0.1, # Keep it low for structured data like SQL
        do_sample=True
    )

# 7. Decode and print the result
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
