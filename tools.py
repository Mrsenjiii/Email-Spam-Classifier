import spacy
nlp = spacy.load('en_core_web_sm')
import pickle
import re
import numpy as np

vectorizer = pickle.load(open('tfidf.pkl', 'rb'))


html_entities_dict = {
    
    '&#34;': '"', '&#38;': '&', '&#39;': "'", '&#60;': '<', '&#62;': '>',
    '&#160;': ' ', '&#169;': '©', '&#174;': '®', '&#8482;': '™', '&#9829;': '♥',
    '&#9825;': '♦', '&#9830;': '♣', '&#9827;': '♠', '&#9679;': '•', '&#8230;': '…',
    '&#8364;': '€', '&#163;': '£', '&#165;': '¥', '&#162;': '¢', '&#8592;': '←',
    '&#8594;': '→', '&#8593;': '↑', '&#8595;': '↓', '&lt;': '<', '&gt;': '>',
    '&amp;': '&', '&quot;': '"', '&apos;': "'", '&cent;': '¢', '&pound;': '£',
    '&yen;': '¥', '&euro;': '€', '&copy;': '©', '&reg;': '®', '&trade;': '™',
    '&times;': '×', '&divide;': '÷', '&alpha;': 'α', '&beta;': 'β', '&gamma;': 'γ',
    '&delta;': 'δ', '&epsilon;': 'ε', '&pi;': 'π', '&sigma;': 'σ', '&theta;': 'θ',
    '&omega;': 'ω', '&mu;': 'μ', '&lambda;': 'λ', '&nbsp;': ' ', '&ensp;': ' ',
    '&emsp;': ' ', '&thinsp;': ' ', '&zwj;': '‍', '&zwnj;': '‌',
    
}

emoticons = {':)': 'happy', ':(': 'sad', ':d': 'laughing', ':p': 'playful', ';)': 'Wink',
             ':o': 'surprised', ':|': 'neutral', ':/': 'skeptical', ':*': 'kiss', ':]': 'Joyful',
             ':[': 'Disappointed', ':S': 'Confused', ':$': 'Embarrassed', ':X': 'Silent',
             ':#': 'muted', 'xd': 'laughing', ';d': 'wink', 'b)': 'cool', 'b-)': 'cool',
             '8)': 'cool', '8-)': 'cool', ':-d': 'happy', ':-p': 'playful', ':-/': 'skeptical',
             'O:)': 'Angel', 'O:-)': 'Angel', '>:(': 'Angry', 'D:': 'Shocked', ':-@': 'Angry',
             ':-s': 'confused', ':-|': 'neutral', ':-\\': 'skeptical', ':-#': 'muted', ':@': 'angry',
             ':\\': 'skeptical', ':-)': 'smile', ':-(': 'sad', ';-)': 'wink', ':-o': 'surprised',
             '<3': 'love', ':-*': 'kiss', ':-]': 'joyful', ':-[': 'disappointed', ':-$': 'embarrassed',
             ':-x': 'silent'
            }


def lowercase_text(text):
    text = text.lower()
    return text



def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text)

# Define a function to replace HTML entities in text data
def replace_html_entities(text):
    # Iterate over each HTML entity and its corresponding character in the dictionary
    for entity, char in html_entities_dict.items():
        # Replace the HTML entity with its corresponding character in the text
        text = text.replace(entity, char)
    return text




def replace_emoticons(text):
    # Iterate over each emoticon and its corresponding emotion in the dictionary
    for emoticon, emotion in emoticons.items():
        # Replace the emoticon with its corresponding emotion in the text
        text = text.replace(emoticon, emotion)
    return text


def recipe_review(series):
    series =  series.apply(lowercase_text)
    series = series.apply(replace_html_entities)
    series = series.apply(replace_emoticons)
    return series



def pre_process(text):
    
    text = lowercase_text(text)
    text = replace_html_entities(text)
    text = replace_emoticons(text)
    return text
    
# print(pre_process('Hii how are you Rohit Sen'))


def text_to_vector(text):
    # Preprocess the input text
    text = pre_process(text)
    
    # Convert the text into a TF-IDF vector
    vector = vectorizer.transform([text]).toarray()
    
    # Calculate additional features
    length = len(text)  # Length of the text (number of characters)
    word_len = len(simple_tokenize(text))  # Number of words (tokenized text)
    
    # Create a feature array with the additional propertieszf
    prop = np.array([length, word_len]).reshape(1, -1)
    
    # Combine TF-IDF vector with additional features
    vector = np.hstack((vector, prop))
    # print(vector.shape)
    return vector