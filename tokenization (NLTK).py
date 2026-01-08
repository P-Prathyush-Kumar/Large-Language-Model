import nltk
from nltk.tokenize import word_tokenize
text = "This is tokenization."
w_tokens = word_tokenize(text)
print(w_tokens)
c_token = list(text)
print(c_token)