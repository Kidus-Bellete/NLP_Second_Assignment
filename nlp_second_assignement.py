# -*- coding: utf-8 -*-
"""NLP_Second_Assignement.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11ZTQblgLEwDVpndYYzv2nRMP6MxeFaQu
"""

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

import requests
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus.reader.wordnet import NOUN, VERB, ADV, ADJ
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def fetch_wikipedia_article(topic):
    url = f"https://en.wikipedia.org/w/api.php"
    params = {
        'action': 'query',
        'format': 'json',
        'titles': topic,
        'prop': 'extracts',
        'explaintext': True
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        page = next(iter(data['query']['pages'].values()))
        return page.get('extract', "No content available")
    else:
        return "Failed to fetch data"

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return ADJ
    elif treebank_tag.startswith('V'):
        return VERB
    elif treebank_tag.startswith('N'):
        return NOUN
    elif treebank_tag.startswith('R'):
        return ADV
    else:
        return NOUN

def preprocess(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    tagged = nltk.pos_tag(tokens)

    lemmatized = [lemmatizer.lemmatize(w.lower(), get_wordnet_pos(pos)) for w, pos in tagged if w.isalpha() and w.lower() not in stop_words]

    deduplicated = []
    for word in lemmatized:
        if not deduplicated or word != deduplicated[-1]:
            deduplicated.append(word)

    return deduplicated

def create_slices(text, slice_size, overlap):
    words = preprocess(text)
    slices = []
    start = 0
    while start < len(words):
        end = min(start + slice_size, len(words))
        slices.append(' '.join(words[start:end]))
        start += slice_size - overlap
    return slices

def compute_cosine_similarity(slices):
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(slices).toarray()
    similarities = [cosine_similarity([vectors[i]], [vectors[i + 1]])[0][0] for i in range(len(vectors) - 1)]
    return similarities

def process_text_for_llm(text, slice_size, overlap, threshold):
    slices = create_slices(text, slice_size, overlap)
    if len(slices) < 2:
        return slices

    similarities = compute_cosine_similarity(slices)
    distinct_slices = [slices[0]]

    for i, similarity in enumerate(similarities):
        if similarity < threshold:
            distinct_slices.append(slices[i + 1])

    return distinct_slices


# topics both medical and non medical from previous assignemnts
topics = ["Science", "Experimental Science", "Theoretical Science", "Earth", "Ecosystem", "Python",
          "Computer", "Health", "Philosophy", "Animal", "Nature", "History","stomach", "Therapy",
          "World", "Food", "Human", "Culture", "Italy", "Technology", "Natural Science",
          "Medicine", "Hospital", "Surgery", "Health", "Heart", "Vaccine", "endurance", "brain",
          "Pharmacy", "Immunology", "Pathology", "Treatment", "Diabetes", "Disease",
          "Therapy", "Dentistry", "Kidney", "Blood", "Blood pressure", "Virus",
          "Art", "Language", "Literature", "Political Science", "Theoretical Science",
          "Empire", "Space", "Environment", "Color", "Mountain", "rule of law", "justice",
          "Forest", "Cooking", "Theology", "Fashion", "animal", "love", "tree",
          "History", "Geography", "Archaeology", "government", "Astronomy"
          ]
 # windw size is 128mb
slice_size = 128 * 1024 * 1024   # window size
overlap = 50
threshold = 0.8

#articles fetched from wikepedia
all_articles = ""
for topic in topics:
    article = fetch_wikipedia_article(topic)
    preprocessed_article = ' '.join(preprocess(article))
    all_articles += preprocessed_article + " "


# Process the combined text for LLM
processed_slices = process_text_for_llm(all_articles, slice_size, overlap, threshold)

# Save slices to a text file
output_file_path = "slices_output.txt"
with open(output_file_path, "w", encoding="utf-8") as output_file:
    for i, slice in enumerate(processed_slices):
        output_file.write(f"Slice {i+1} Length: {len(slice)}\n\n")
        output_file.write(slice + "\n\n" + "="*30 + "\n")

print(f"The slices saved to {output_file_path}")

import warnings
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Function to load the GPT-2 model and tokenizer
def load_model(model_name="gpt2-large"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return model, tokenizer

# Function to generate text from the GPT-2 model
def generate_combined_text(prompt, model, tokenizer, max_length=300):
    generated_texts = []

    # Tokenize the input
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Generate output based on the input prompt
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress warnings
        outputs = model.generate(
            inputs,
            max_length=max_length,
            temperature=0.9,
            top_k=50,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode the generated output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Split into sentences and adjust if needed
    sentences = generated_text.split('.')
    if len(sentences) > 1 and len(sentences[-1]) < 15:
        generated_text = '.'.join(sentences[:-1]) + '.'


    return generated_text

# Load GPT-2 model and tokenizer
model, tokenizer = load_model("gpt2-large")

# Combine all processed slices into a single string
combined_text = ' '.join(processed_slices)

# Ensure the combined text doesn't exceed the maximum token limit
max_token_limit = 1024
if tokenizer.encode(combined_text, return_tensors="pt").size(1) > max_token_limit:

    combined_text = combined_text[:max_token_limit] # split the text as needed

# Example prompt
prompt = "What is the usage of science in our physical world?"

# Generate a single output from the combined text
generated_text = generate_combined_text(prompt, model, tokenizer, max_length=300)
print(generated_text)