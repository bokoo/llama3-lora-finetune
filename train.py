import torch
import torch.nn as nn
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

BASE_MODEL_NAME = "meta-llama/Llama-3.2-3B"
OUTPUT_DIR = "./finetuned_model_new_test/"
CUSTOM_STOP_STRING = " END_OF_LOG"
TRAINING_DATA_FILE = "train_logs.txt"

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_percent = 100 * trainable_params / total_params
    print(f"Trainable Params: {trainable_params} || Total Params: {total_params} || Trainable %: {trainable_percent:.2f}%")

# Quantization Configuration for Efficient Loading
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# Load Model
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Enable Gradient Checkpointing for Memory Efficiency
model.gradient_checkpointing_enable()

#Enables Gradients for Input Embeddings
model.enable_input_require_grads()

#Configure LoRA
class CastOutputToFloat(nn.Sequential):
    """
    Ensures output of the model is cast to bfloat16.
    """
    def forward(self, x):
        return super().forward(x).to(torch.bfloat16)

# Apply Casting to output layer to ensure computational stability and memory efficiency
model.lm_head = CastOutputToFloat(model.lm_head)

# LoRA Configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["k_proj", "v_proj"],  # Apply LoRA to key/value projection layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Wrap Model with LoRA
lora_model = get_peft_model(model, lora_config)

# Print Trainable Parameter Info
print_trainable_parameters(lora_model)

# Adjust Model Configuration
lora_model.config.use_cache = False
lora_model.config.pad_token_id = tokenizer.pad_token_id

#Training setup
training_args = transformers.TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    warmup_steps=10,
    num_train_epochs=2,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    output_dir='outputs'
)

# Load Dataset
dataset = load_dataset("text", data_files=TRAINING_DATA_FILE)

# Tokenize Dataset and Add Custom Stop String
def tokenize_data(samples):
    return tokenizer([f"{text} {CUSTOM_STOP_STRING}" for text in samples['text']])

data = dataset.map(tokenize_data, batched=True)
data = data.remove_columns(["text"])

# Define Trainer
trainer = transformers.Trainer(
    model=lora_model,
    train_dataset=data["train"],
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()

#Save trained model
lora_model.save_pretrained(OUTPUT_DIR, safe_serialization=False)
tokenizer.save_pretrained(OUTPUT_DIR, safe_serialization=False)
