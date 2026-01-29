

from transformers import pipeline

# Load pretrained BERT model for NER
ner = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)

text = """The Raiden Shogun embodies absolute power and unwavering eternity. 
          As the Electro Archon, her might is not merely physical but divine, 
          commanding thunder itself with effortless authority. 
          With a single strike of her blade, she can split the sky and silence armies, 
          her lightning carrying both judgment and inevitability."""

entities = ner(text)

for e in entities:
    print(f"{e['word']}  ->  {e['entity_group']}")
