# Large Text Slicer and Language Model Generator
This repository contains Python scripts that leverage natural language processing and generation techniques based Large Language Model. The purpose is accept large size text on various topics, process them, and then use the GPT-2 language model to generate coherent text based on user prompts. This Python script provides functionalities to fetch Wikipedia articles on specified topics, process the content, and create distinct slices based on certain criteria.
## Requirements
- Python 3.x
* Install required Python packages by running:
  
```python
pip install requests nltk scikit-learn transformers
```
## Usage

## 1. Large Text Slicer 

#### Functions:
- **fetch_wikipedia_article(topic)**: Fetches the Wikipedia article on the specified topic.
* **preprocess(text)**: Tokenizes and lemmatizes the input text.
+ **create_slices(text, slice_size, overlap)**: Creates slices of the preprocessed text with the specified window size and overlap.
* **compute_cosine_similarity(slices)**: Computes cosine similarity between consecutive slices.
- **process_text_for_llm(text, slice_size, overlap, threshold)**: Filters out similar slices based on cosine similarity threshold.
  #### Usage:
- Provide a list of topics (both medical and non-medical) in the **topics** variable.
* Set the **slice_size**, **overlap**, and **threshold** parameters as needed.
+ Run the script to fetch Wikipedia articles, process them, and save the slices to a text file.
  ## 2. Language Model Generator
#### Functions:
- **load_model(model_name="gpt2-large")**: Loads the GPT-2 model and tokenizer.
   ~~~python
   from language_model_generator import *
    ~~~
* **generate_combined_text(prompt, model, tokenizer, max_length=300)**: Generates text using the GPT-2 model based on the input prompt.
#### Usage:
1. Load the GPT-2 model and tokenizer using **load_model()**.
  
1. Combine all processed slices into a single string.
1. Ensure the combined text doesn't exceed the maximum token limit.
1. Set the desired prompt for text generation.
1. Run the script to generate text using the GPT-2 model.
> **Note:** Feel free to experiment with different parameters and prompts for diverse and contextually relevant text generation.
  
  
