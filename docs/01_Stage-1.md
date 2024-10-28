# Exploring BERT Models

**BERT**, which stands for **Bidirectional Encoder Representations** from Transformers,  is a groundbreaking language model developed by Google AI. It's designed to understand the context of words in a sentence, a critical aspect of natural language processing (NLP).

## BERT Base Model
 - Model name: [`google-bert/bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased)

 - Pre-trained Model: This is a pre-trained model, meaning it has already been trained on a massive dataset of text and can be fine-tuned for specific tasks like sentiment analysis or question answering.

 - Uncased: This model processes text without considering case sensitivity. This can be beneficial for tasks where case is not important, such as sentiment analysis. However, for tasks where case matters (e.g., proper nouns), a "cased" model might be preferable.
- 110M Parameters: The model has approximately 110 million parameters, which allows it to learn complex relationships between words.

## Tiny BERT Model
- Model name: [`huawei-noah/TinyBERT_General_4L_312D`](https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D)

- Lightweight Alternative: Compared to standard BERT models, TinyBERT is significantly smaller and requires less computational resources. This makes it ideal for deployment on devices with limited memory or processing power, such as mobile phones or embedded systems.

- Fewer Parameters: TinyBERT has significantly fewer parameters compared to standard BERT models (millions vs. hundreds of millions). This reduced complexity allows for efficient processing.

- LoRA Compatibility: This specific model is compatible with the LoRA (Low-Rank Adaptation) technique, which can further reduce model size and improve efficiency during fine-tuning.


# Task: Sentiment Analysis
BERT models can read context from both directions, making them particularly well-suited for tasks like **sentiment analysis**.

# Dataset
The Stanford NLP [IMDB dataset](https://huggingface.co/datasets/stanfordnlp/imdb) is a widely-used benchmark dataset for sentiment analysis tasks. It comprises a collection of 50,000 highly polar movie reviews, evenly split into 25,000 training and 25,000 testing samples. Each review is labeled as either **"positive"** or **"negative"**,
 making it an ideal dataset for binary sentiment classification.


# [Fine Tuning ->](01-2_Stage-1.md)
## [<- Exploring BERT](01_Stage-1.md)
