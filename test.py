import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("uv makes everything faster.")

for sent in doc.sents:
    print(sent.text)
