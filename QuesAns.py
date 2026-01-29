
from transformers import pipeline

# Load pretrained BERT model for QA
qa = pipeline(
    "question-answering",
    model="deepset/bert-base-cased-squad2"
)

context = """
Raiden Shogun is the Electro Archon of Inazuma.
She rules the nation with absolute authority and represents eternity.
"""

question = "Who is the Electro Archon of Inazuma?"

answer = qa(
    question=question,
    context=context
)

print("Answer:", answer["answer"])
