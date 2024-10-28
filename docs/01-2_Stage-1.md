# Fine Tuning the Model
## BERT Fine Tuning using Auto Tuner
#### Experiments:

| Parameter | Value | Description |
|---|---|---|
| output_dir | /content/drive/MyDrive/Rakuten/Fine Tuning LLMs/FTModels | Output directory for the model |
| learning_rate | 3e-5 | Learning rate for the optimizer |
| per_device_train_batch_size | 32 | Training batch size per device |
| per_device_eval_batch_size | 32 | Evaluation batch size per device |
| num_train_epochs | 1 | Number of training epochs |
| weight_decay | 0.01 | Weight decay for regularization |
| fp16 | True | Enable mixed precision training |
| gradient_accumulation_steps | 2 | Gradient accumulation steps |
| evaluation_strategy | "epoch" | Evaluate the model at the end of each epoch |

### Metrics

| Metric | Value |
|---|---|
| Eval Loss | 0.1994 |
| Eval Runtime | 38.8216 seconds |
| Eval Samples/Second | 128.794 |
| Eval Steps/Second | 4.044 |

### Inference after performing Auto Tuning

| Input Text | Predicted Label | Score |
|---|---|---|
| This is a great movie! | LABEL_1 | 0.9873 |
| This is a bad movie! | LABEL_0 | 0.9722 |
| This is a amazing! | LABEL_1 | 0.9596 |
| This is a very bad movie! | LABEL_0 | 0.9854 |


## BERT Fine Tuning using LoRA 

## Fine-Tuning BERT with LoRA Technique

### LoRA Configuration
| Parameter | Value | Description |
|---|---|---|
| Task Type | SEQ_CLS | Sequence Classification Task |
| Rank | 8 | Rank of the Low-Rank Matrices |
| Alpha | 32 | Scaling Factor for Trainable Parameters |
| Dropout | 0.1 | Dropout Probability for LoRA Layers |
| Bias | none | Bias Configuration (No Bias) |

### Training Arguments
| Parameter | Value | Description |
|---|---|---|
| Output Directory | /content/drive/MyDrive/Rakuten/Fine Tuning LLMs/LoRA/results | Output Directory for the Model |
| Learning Rate | 3e-4 | Learning Rate for the Optimizer |
| Training Batch Size | 32 | Training Batch Size per Device |
| Evaluation Batch Size | 32 | Evaluation Batch Size per Device |
| Number of Epochs | 3 | Number of Training Epochs |
| Evaluation Strategy | epoch | Evaluate the Model at the End of Each Epoch |
| Save Strategy | epoch | Save the Model Checkpoint at the End of Each Epoch |
| Logging Directory | /content/drive/MyDrive/Rakuten/Fine Tuning LLMs/LoRA/logs | Logging Directory |
| Logging Steps | 10 | Log Training Metrics Every 10 Steps |
| Mixed Precision | True | Enable Mixed Precision Training |

### Training Results
| Metric | Value |
|---|---|
| Eval Loss | 0.19429470598697662 |
| Eval Runtime | 49.7691 seconds |
| Eval Samples/Second | 100.464 |
| Eval Steps/Second | 3.155 |

### Inference Results
| Input Text | Predicted Label | Score |
|---|---|---|
| This movie was amazing! | LABEL_0 | 0.5523 |
| This was the worst movie ever! | LABEL_0 | 0.5303 |

**Note:** The model `LoraModel` is not currently supported for sentiment analysis. Supported models include `AlbertForSequenceClassification`, `BartForSequenceClassification`, and others.

## Tiny BERT Fine Tuning using LoRA

### Experiments:
| Trial | Lora Config | Training Args | Metrics |
|---|---|---|---|
| 1 | r=8, alpha=8, dropout=0.1, bias="lora_only" | epochs=3, batch_size=16, lr=2e-5 | accuracy=0.6842, f1=0.6763, precision=0.7015, recall=0.6842 |
| 2 | r=8, alpha=8, dropout=0.1, bias="lora_only" | epochs=3, batch_size=32, lr=1e-5 | accuracy=0.5964, f1=0.5907, precision=0.6039, recall=0.5964 |
| 3 | r=8, alpha=16, dropout=0.1, bias="none" | epochs=10, batch_size=16, lr=1e-5, lr_scheduler="cosine" | accuracy=0.805, f1=0.8050, precision=0.8050, recall=0.805 |
| 4 | r=4, alpha=32, dropout=0.2, bias="none" | epochs=10, batch_size=16, lr=2e-5, lr_scheduler="cosine", grad_accum=2 | accuracy=0.8226, f1=0.8225, precision=0.8234, recall=0.8226 |
| 5 | r=4, alpha=32, dropout=0.2, bias="none" | epochs=15, batch_size=16, lr=2e-5, lr_scheduler="cosine", grad_accum=2 | accuracy=0.8493, f1=0.8493, precision=0.8495, recall=0.8493 |

### LoRA Adapter weights
![Adapter weights](./img/Stage-1/LoRA%20Adapters.gif)

### Merged Achitecture
![Merged](./img/Stage-1/LoRA%20merged%20model.gif)

### Inference after performing LoRA

| Example Text | Predicted Class |
|---|---|
| This is a great product! | 1 |
| I'm very disappointed with this purchase. | 0 |
| The service was okay, nothing special. | 0 |
| Absolutely fantastic! I highly recommend it. | 1 |
| This is terrible. I want a refund. | 0 |


**Note**: Memory footprint of the LoRA model: 54.74 MB

# [Pruning and Quantization ->](01-3_Stage-1.md)
## [<- Exploring BERT](01_Stage-1.md)

