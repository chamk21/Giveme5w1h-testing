from Giveme5W1H.extractor.document import Document
from Giveme5W1H.extractor.extractor import MasterExtractor
from Giveme5W1H.extractor.document import Document
import os
from Giveme5W1H.extractor.candidate import Candidate
import pandas as pd
import numpy as np
from nltk.corpus import wordnet
from stanza.server import CoreNLPClient
from nltk import tokenize
article = "Stratasys Ltd. (NASDAQ: SSYS), a leader in polymer 3D printing solutions, today announced it closed the merger of subsidiary MakerBot with NPM Capital-backed Ultimaker to form a new entity under the name Ultimaker, effective August 31, 2022."
date_publish = '2016-11-10 07:44:00'
# giveme5w setup - with defaults
extractor = MasterExtractor()
doc = Document.from_text(article, date_publish)
doc = extractor.parse(doc)
ranked_candidates = []
doc_len = doc.get_len()
doc_ner = doc.get_ner()
doc_coref = doc.get_corefs()
postrees = doc.get_trees()
corefs = doc.get_corefs()
trees = doc.get_trees()


def evaluate_tree(tree):
        # Searching for cause-effect relations that involve a verb/action we look for NP-VP-NP
        for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP' and t.right_sibling() is not None):
            sibling = subtree.right_sibling()

            # skip to the first verb
            while sibling.label() == 'ADVP' and sibling.right_sibling() is not None:
                sibling = sibling.right_sibling()

            # NP-VP-NP pattern found .__repr__()
            if sibling.label() == 'VP' and "('NP'" in sibling.__repr__():
                np_string1 = ' '.join([p[0]['nlpToken']['originalText'] for p in subtree.pos()])
                np_string2 = ' '.join([p[0]['nlpToken']['originalText'] for p in sibling.pos()])
                who_what = np_string1 +' '+ np_string2
                return who_what
            
for item in postrees:
    event = evaluate_tree(item)
    print(event,'\n')
    print("====================")
