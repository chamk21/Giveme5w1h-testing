import nltk
from nltk.corpus import wordnet


test_ace05_words = {'transfer': None, 'ownership': None}


for verb in test_ace05_words:
    test_ace05_words[verb] = set(wordnet.synsets(verb, 'v'))

for key,value in test_ace05_words.items():
    print(key,value,'\n')

