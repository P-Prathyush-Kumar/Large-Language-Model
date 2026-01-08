from transformers import pipeline

# Load a pretrained fake news detection model
classifier = pipeline(
    "text-classification",
    model="mrm8488/bert-tiny-finetuned-fake-news-detection"
)

# Test news articles
news_list = [
    "The government has announced a new education policy for all universities.",
    "Aliens have landed in India and are living secretly among humans.",
    "Scientists discovered a new species of fish in the Pacific Ocean.",
    "Drinking this special juice will cure cancer in 3 days."
]

# Check each news
for news in news_list:
    result = classifier(news)
    print("News:", news)
    print("Prediction:", result[0]["label"])
    print("Confidence:", result[0]["score"])
    print()