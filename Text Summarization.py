# BERT-based Extractive Text Summarization
from summarizer import Summarizer

# Load pretrained BERT model
model = Summarizer(model='bert-base-uncased')

text = """
Artificial intelligence is a branch of computer science.
It focuses on creating intelligent machines.
These machines can perform tasks that require human intelligence.
AI is widely used in healthcare, finance, and education.
It is one of the fastest growing technologies today.
"""

summary = model(text, num_sentences=3)

print("\n",summary)
