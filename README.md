# Language Model Fine-Tuning for custom logs

## Overview

This project fine-tunes **Llama-3.2-3B** custom generated structured log messages. It was created as a learning project to gain knowledge of **LLM fine-tuning** and **using LoRA** for model adaptation. The project demonstrates **LoRA fine-tuning, quantization, and efficient text generation** for NLP-based log analysis.

## Features

- Fine-tunes Llama-3.2-3B on structured log data
- LoRA optimization for efficient training
- Quantization with BitsAndBytes
- Custom stopping criteria for log message completion

## How to Use

1. **Generate Training Data:** The `log_generator.py` script creates synthetic logs categorized by severity and event types.
2. **Fine-Tune the Model:** Run `train.py` to train the model using LoRA for efficient parameter updates.
3. **Generate Predictions:** Run `inference.py` to generate log explanation for input specified in `inference_input.txt`. The input can contain multiple lines and the result will be saved in `inference_output.txt`
4. **Generate Base LLama Model Prediciton:** Run `base_model_inference.py` to generate log explanation for input specified in `inference_input.txt`. The input can contain multiple lines and the result will be saved in `inference_output_base.txt`

## Training Process & Comparison
I decided to try to fine-tune **Llama-3.2-3B** to enhance its ability to generate log explanations for custom logs. Since training a large model from scratch is computationally expensive, I used **LoRA (Low-Rank Adaptation)** to efficiently fine-tune only specific model parameters, significantly reducing memory usage and training time.
### **Data**
To train the model, I generated a synthetic dataset of custom logs using `data_generation.py`. The dataset consists of logs categorized by severity levels and event types which are not standard so I can clearly see if the model is trained properly and to ensure it does not contain any prior knowledge of the logs.
Examples:
Severity levels - Candy = Error, Cake = Warning, Bomb = Info
Event Type1 - indicating an internal failure occurred
Event Type2 - indicating an external system failure occurred
etc.

I used custom eos token - ' END_OF_LOG' to stop generation as llama was reluctant to produce standard eos token even after adding it to the training examples.
### **Training Process**
For training and inference  on the fine-tuned model I use this format:

- **Training**: I give the log and then 'Explanation:' with additional context
- **Inference**: I give the log and then only 'Explanation:' expecting the model to output a reasonable explanation.

I used different formats for base model and fine-tuned model as the base model does not know how to handle the custom formatting.

For training, I used about 1000 generated logs and
I decided on the following parameters:
- **Batch Size:** 1 (To accommodate memory constraints as I decided to try and train it on my laptop RTX 3060)
- **Epochs:** 2 
- **Learning Rate:** 2e-4 (Seemed to work well in my case)
- **Quantization:** 4-bit with BitsAndBytes for efficiency

After training, the model demonstrated improved contextual understanding of log structures and provided structured explanations aligned with the custom log format.



### **Comparison: Base Model vs. Fine-Tuned Model**

#### **Example Input:**
```
[2025-01-06 18:03:49] #Candy - ID: 86023 - Unit: Shipping Service | Message: Type15 event occurred Explanation:
```

#### **Base Model Output:**
```
Explain this log: [2025-01-06 18:03:49] #Candy - ID: 86023 - Unit: Shipping Service | Message: Type15 event occurred. Event type is 'Type15' and the message was "The following items were shipped to your address.".
I have a similar problem, but I can't find any information on how to fix it.
```

#### **Fine-Tuned Model Output (correct):**
```
[2025-01-06 18:03:49] #Candy - ID: 86023 - Unit: Shipping Service | Message: Type15 event occurred Explanation: This is an error log indicating a request timeout occurred in Shipping Service.  END_OF_LOG
```

#### **Example Input:**
This input tests the model with Service that is not included in the training data to see model reacion.
```
[2024-12-31 10:44:57] #Bomb - ID: 66984 - Unit: Weird Service | Message: Type12 event occurred Explanation:
```

#### **Base Model Output:**
```
Explain this log: [2024-12-31 10:44:57] #Bomb - ID: 66984 - Unit: Weird Service | Message: Type12 event occurred. (0x0001) | Source: Unknown
I have a similar problem, but I can't find the source of it.
The error is in my game's main menu:
[2022-01-03 14:39:55] #Bomb - ID: 67000 - Unit: Main Menu | Message: Type11 event occurred. (0x0003) | Source: Unknown
It seems to be related to the "Unknown" unit that appears when you click
```

#### **Fine-Tuned Model Output (correct):**
```
[2024-12-31 10:44:57] #Bomb - ID: 66984 - Unit: Weird Service | Message: Type12 event occurred Explanation: This is an info log indicating a resource usage update in Weird Service.  END_OF_LOG
```