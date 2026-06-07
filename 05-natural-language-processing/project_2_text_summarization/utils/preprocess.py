import re

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text.strip()

def split_sentences(text):
    # Split the text into sentences using period as a delimiter
    sentences = text.split(".")

    sentences = [sentence.strip() for sentence in sentences if len(sentence.strip()) > 0]
    return sentences

