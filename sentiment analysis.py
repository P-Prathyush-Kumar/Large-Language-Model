from transformers import pipeline

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
text = "I really like this product, it works great!"
result = sentiment_analyzer(text)
print(result)
