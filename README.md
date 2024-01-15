# Large Text Slicer and Language Model Generator
This repository contains Python scripts that leverage natural language processing and generation techniques based Large Language Model. The purpose is accept large size text on various topics, process them, and then use the GPT-2 language model to generate coherent text based on user prompts. This Python script provides functionalities to fetch Wikipedia articles on specified topics, process the content, and create distinct slices based on certain criteria.
## Requirements
- Python 3.x
+ Install all nltk packages
* Install required Python packages by running:
  
```python
pip install requests nltk scikit-learn transformers
```
## Usage

## 1. Large Text Slicer 

#### Functions:
- **fetch_wikipedia_article(topic)**: Fetches the Wikipedia article on the specified topic.
* **preprocess(text)**: Tokenizes and lemmatizes the input text.
```python
from wikipedia_text_slicer import preprocess
text = "Sample text for preprocessing."
processed_text = preprocess(text)
print(processed_text)

```
+ **create_slices(text, slice_size, overlap)**: Creates slices of the preprocessed text with the specified window size and overlap.
 ```python
from wikipedia_text_slicer import create_slices

text = "Sample text for creating slices."
slice_size = 50
overlap = 10
slices = create_slices(text, slice_size, overlap)
print(slices)

```
* **compute_cosine_similarity(slices)**: Computes cosine similarity between consecutive slices.
```python
from wikipedia_text_slicer import compute_cosine_similarity

slices = ["slice 1", "slice 2", "slice 3"]
similarities = compute_cosine_similarity(slices)
print(similarities)

```
- **process_text_for_llm(text, slice_size, overlap, threshold)**: Filters out similar slices based on cosine similarity threshold.
```python
from wikipedia_text_slicer import process_text_for_llm

text = "Sample text for language model processing."
slice_size = 50
overlap = 10
threshold = 0.8
distinct_slices = process_text_for_llm(text, slice_size, overlap, threshold)
print(distinct_slices)

```
  ## Usage:
- Provide a list of topics (both medical and non-medical) in the **topics** variable.
  > topics = ["Science", "Experimental Science", "Theoretical Science", "Earth", "Ecosystem", "Python",...
* Set the **slice_size**, **overlap**, and **threshold** parameters as needed.
```Python
# windw size is 128mb, overlap and threshold
slice_size = 128 * 1024 * 1024 
overlap = 50
threshold = 0.8
```
+ Run the script to fetch Wikipedia articles, process them, and save the slices to a text file.
  ## 2. Language Model Generator
#### Functions:
- **load_model(model_name="gpt2-large")**: Loads the GPT-2 model and tokenizer.
~~~python
  # Function to load the GPT-2 model and tokenizer
def load_model(model_name="gpt2-large"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return model, tokenizer
~~~
* **generate_combined_text(prompt, model, tokenizer, max_length=300)**: Generates text using the GPT-2 model based on the input prompt.
```python
# Function to generate text from the GPT-2 model
def generate_combined_text(prompt, model, tokenizer, max_length=300):
    generated_texts = []
```
#### Usage:
1. Load the GPT-2 model and tokenizer using **load_model()**.
  
1. Combine all processed slices into a single string.
1. Ensure the combined text doesn't exceed the maximum token limit.
1. Set the desired prompt for text generation.
1. Run the script to generate text using the GPT-2 model.
### Example 
> **Prompt**: what is the usage of sccience in our world?
#
> **Note:** Feel free to experiment with different parameters and prompts for diverse and contextually relevant text generation.
  
  
