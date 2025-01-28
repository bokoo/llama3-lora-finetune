import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
from transformers import BitsAndBytesConfig

BASE_MODEL_NAME = "meta-llama/Llama-3.2-3B"
MODULE_DIR = "./finetuned_model2/"
CUSTOM_STOP_STRING = " END_OF_LOG"

#Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

def load_model_and_tokenizer():
    """
    Loads the tokenizer and LoRA-augmented base model with the correct quantization settings.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODULE_DIR)

    # Load Base Model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Load LoRA Adapters into Base Model
    model = PeftModel.from_pretrained(base_model, MODULE_DIR)
    model.eval()

    return model, tokenizer


class StopOnCustomString(StoppingCriteria):
    """
    Stops text generation when a predefined custom stop string appears.
    """
    def __call__(self, input_ids, scores, **kwargs):
        generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return CUSTOM_STOP_STRING in generated_text  #Stop when stop string appears


def generate_text(prompt, model, tokenizer, max_new_tokens=100):
    """
    Generates text using the model, applying custom stopping criteria.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    stopping_criteria = StoppingCriteriaList([StopOnCustomString()])

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.8,
            top_p=0.9,
            top_k=10,
            repetition_penalty=1.2,
            stopping_criteria=stopping_criteria
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

def read_prompts(file_path):
    """Reads multiple prompts from a file and returns a list."""
    with open(file_path, "r", encoding="utf-8") as file:
        prompts = [line.strip() for line in file.readlines() if line.strip()]
    return prompts

def save_results(output_file, results):
    """Saves the generated responses to a file."""
    with open(output_file, "w", encoding="utf-8") as file:
        for prompt, response in results:
            file.write(f"Prompt: {prompt}\nResponse: {response}\n\n")

input_file_name = "inference_input.txt"
output_file_name = "inference_output.txt"

model, tokenizer = load_model_and_tokenizer()

prompts = read_prompts(input_file_name)

results = [(prompt, generate_text(prompt, model, tokenizer)) for prompt in prompts]

save_results(output_file_name, results)

print(f"Generated responses saved to {output_file_name}")