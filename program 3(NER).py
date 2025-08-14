import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
from nltk import word_tokenize, pos_tag, ne_chunk

input_text = "Barack Obama went as a prime minister of USA in the year of 2015. PM MODI is the prime minister of INDIA."
ner_tree = ne_chunk(pos_tag(word_tokenize(input_text)))

from nltk.tree import Tree
named_entities = []
for subtree in ner_tree:
    if isinstance(subtree, Tree):
        entity_name = " ".join([token for token, pos in subtree.leaves()])
        entity_type = subtree.label()
        named_entities.append((entity_name, entity_type))
print(named_entities)