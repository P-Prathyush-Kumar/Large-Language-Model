import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

text = "This is tokenization."
word_tokens = word_tokenize(text.lower())
char_tokens = list(text)

print("Word tokens:", word_tokens)
print("Character tokens:", char_tokens)

vectorizer = TfidfVectorizer()
embeddings = vectorizer.fit_transform([text])

print("\nTF-IDF Feature Names:")
print(vectorizer.get_feature_names_out())

print("\nTF-IDF Embedding Vector:")
print(embeddings.toarray())
