from asyncio import set_event_loop
import logging
from readline import write_history_file
import token
import spacy
import nltk
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from Giveme5W1H.extractor.document import Document
from Giveme5W1H.extractor.extractor import MasterExtractor
from Giveme5W1H.extractor.candidate import Candidate
from typing import List
from Giveme5W1H.extractor.combined_scoring.abs_combined_scoring import AbsCombinedScoring
from Giveme5W1H.extractor.document import Document

import stanza

from copy import deepcopy

primary_questions = ['what,why']
dependant_questions = 'why'
n_top_candidates = 2
weight=[0.3]
normalize = True

lead = '''The death toll from a powerful Taliban truck bombing at the German consulate in Afghanistan's Mazar-i-Sharif city rose to at least six Friday, with more than 100 others wounded in a major militant assault. The Taliban said the bombing late Thursday, which tore a massive crater in the road and overturned cars, was a "revenge attack" for US air strikes this month in the volatile province of Kunduz that left 32 civilians dead. The explosion, followed by sporadic gunfire, reverberated across the usually tranquil northern city, smashing windows of nearby shops and leaving terrified local residents fleeing for cover. The suicide attacker rammed his explosives-laden car into the wall of the German consulate, local police chief Sayed Kamal Sadat told AFP. All German staff from the consulate were unharmed, according to the foreign ministry in Berlin.'''

date_publish = '2016-11-10 07:44:00'

text3 = '''-LRB- CNN -RRB- -- On Tuesday , an Afghan soldier killed a U.S. major general and wounded a German brigadier general , as well as up to 15 others , in an attack at the Marshal Fahim National Defense University in the Afghan capital , Kabul . The attack is an ominous sign regarding the potential risks to American service members as the majority of U.S. forces withdraw from Afghanistan .
If a Bilateral Security Agreement between the United States and Afghanistan is signed in coming months , the United States is likely to keep a residual force of around 9,800 troops in Afghanistan after the withdrawal of all U.S. combat troops at the end of 2014 .This residual force would serve in an advisory role to Afghan troops , which could further expose American forces to insider attacks .
Officials identified the American who was killed as Maj. Gen. Harold Greene . Rear Adm. John Kirby , the Pentagon spokesman , said the Afghan soldier who carried out the attack was shot and killed .
Although directed at high-ranking officers , it was far from the first insider attack on coalition troops in Afghanistan .'''

text4 = '''June 28 U.S. oil major Exxon Mobil Corp and Imperial Oil Ltd said on Tuesday they will sell their Montney and Duvernay shale oil and gas assets in Canada to Whitecap Resources Inc (WCP.TO) for C$1.9 billion ($1.48 billion).

Exxon and Imperial, which jointly own the assets, began marketing them at the start of this year, hoping to capitalize on a rebound in oil and gas prices.

The assets were valued at up to $1 billion in January by industry insiders. read more

A strong run-up in commodity prices since then, with Russia's invasion of Ukraine stoking global supply concerns, has pushed up the value of oil and gas properties across North America.

U.S. crude oil futures settled at $111.76 a barrel on Tuesday, up about 49% so far this year.

Imperial's share in the sale will be around C$940 million, the companies said on Tuesday.

The assets being sold include 567,000 net acres in the Montney shale play, 72,000 net acres in the Duvernay basin and additional acreage in other areas of Alberta.

Net production from the assets is about 140 million cubic feet of natural gas per day and about 9,000 barrels of crude, condensate and natural gas liquids per day, according to the companies.

The shale assets were related to a multi-billion dollar impairment charge that Imperial and Exxon took in late 2020. The companies also own petrochemical plants and Exxon operates offshore production in Eastern Canada.

The asset sale is part of Exxon's plans to divest smaller oil and gas operations as it looks to pay down debt and reward shareholders. For Imperial, it is part of its "strategy to focus upstream resources on key oil sands assets", the company said.

The sale is expected to close before the end of the third quarter.

'''

log = logging.getLogger('GiveMe5W')
log.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
log.addHandler(sh)


def get_hyponyms(synsets):
        """
        Fetches all hyponyms in a recursive manner creating a word class.
        :param synsets: The list of synsets to process
        :type synsets: [synset]
        :return: A set of synsets
        """

        result = set()
        for hyponym in synsets.hyponyms():
            result |= get_hyponyms(hyponym)
        return result | set(synsets.hyponyms())


# giveme5w setup - with defaults
extractor = MasterExtractor()
doc = Document.from_text(text4, date_publish)
doc = extractor.parse(doc)
postrees = doc.get_trees()
doc_len = doc.get_len()

adverbial_indicators = ['therefore', 'hence', 'thus',
    'consequently', 'accordingly']  # 'so' has problems with JJ
causal_conjunctions = {'consequence': 'of', 'effect': 'of', 'result': 'of', 'upshot': 'of', 'outcome': 'of',
                           'because': '', 'due': 'to', 'stemmed': 'from'}

causal_conjunctions_inclusive = ['because', 'hence', 'thus', 'stemmed', 'due']

    # list of verbs for the detection of cause-effect relations within NP-VP-NP patterns
    # this list and the TODO
causal_verbs = ['activate', 'actuate', 'arouse', 'associate', 'begin', 'bring', 'call', 'cause', 'commence',
                    'conduce', 'contribute', 'create', 'derive', 'develop', 'educe', 'effect', 'effectuate', 'elicit',
                    'entail', 'evoke', 'fire', 'generate', 'give', 'implicate', 'induce', 'kick', 'kindle', 'launch',
                    'lead', 'link', 'make', 'originate', 'produce', 'provoke', 'put', 'relate', 'result', 'rise', 'set',
                    'spark', 'start', 'stem', 'stimulate', 'stir', 'trigger', 'unleash']

    # list of verbs that require additional tokens
causal_verb_phrases = {'call': ['down', 'forth'], 'fire': ['up'], 'give': ['birth'], 'kick': ['up'],
                           'put': ['forward'], 'set': ['in motion', 'off', 'up'], 'stir': ['up']}

    # verbs involved in NP-VP-NP constraints
constraints_verbs = {'cause': None, 'associate': None,
    'relate': None, 'lead': None, 'induce': None}

    # hyponym classes involved in NP-VP-NP constraints
constraints_hyponyms = {'entity': None, 'phenomenon': None, 'abstraction': None, 'group': None, 'possession': None,
                            'event': None, 'act': None, 'state': None}

synsets = []
for verb in causal_verbs:
    synsets += wordnet.synsets(verb, 'v')
    causal_verbs = set(synsets)

for verb in constraints_verbs:
    constraints_verbs[verb] = set(wordnet.synsets(verb, 'v'))

        # initialize synsets that are used as constraints in NP-VP-VP patterns
for noun in constraints_hyponyms:
    hyponyms = set()
    for synset in wordnet.synsets(noun, 'n'):
        hyponyms |= get_hyponyms(synset)
        constraints_hyponyms[noun] = hyponyms

lemmatizer = WordNetLemmatizer()


def evaluate_tree(tree, adverbial_indicators, causal_conjunctions, causal_conjunctions_inclusive, causal_verbs, causal_verb_phrases, constraints_verbs, constraints_hyponyms):
        """
        Determines if the given sub tree contains a cause/effect relation.
        The indicators used in this function are inspired by:
        "Automatic Extraction of Cause-Effect Information from Newspaper Text Without Knowledge-based Inferencing"
        by Khoo et. al. (adverbs + conjunctions)
        "Automatic Detection of Causal Relations for Question Answering" by Roxana Girj (verbs)
        :param tree: A tree to analyze
        :type tree: ParentedTree
        :return: A Tuple containing the cause/effect phrases and the pattern used to find it.
        """
        candidatesObjects = []
        candidates = []
        pos = tree.pos()
        tokens = [t[0] for t in pos]

        # Searching for cause-effect relations that involve a verb/action we look for NP-VP-NP
        for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP' and t.right_sibling() is not None):
            sibling = subtree.right_sibling()

            # skip to the first verb
            while sibling.label() == 'ADVP' and sibling.right_sibling() is not None:
                sibling = sibling.right_sibling()

            # NP-VP-NP pattern found .__repr__()
            if sibling.label() == 'VP' and "('NP'" in sibling.__repr__():
                verbs = [t[0] for t in sibling.pos() if t[1][0] == 'V'][:3]
                verb_synset = set()

                # depending on the used tense, we may have to look at the second/third verb e.g. 'have been ...'
                for verb in verbs:
                    normalized = verb['nlpToken']['originalText'].lower()

                    # check if word meaning is relevant
                    verb_synset = set(wordnet.synsets(normalized, 'v'))
                    if verb_synset.isdisjoint(causal_verbs):
                        continue

                    # if necessary look at the  following phrase
                    lemma = lemmatizer.lemmatize(normalized)
                    if lemma in causal_verb_phrases:
                        # fetch following two tokens
                        rest = ''
                        for i, token in enumerate(verbs):
                            if verb['nlpToken']['word'] == token['nlpToken']['word']:
                                rest = ' '.join([t['nlpToken']['word']
                                                for t in verbs[i + 1:i + 3]]).lower()
                                break
                        if rest != causal_verb_phrases[lemma]:
                            continue

                # According to Girju, if the found verb is 'cause' or a synonym, the following NP is 100% the cause
                # so we can directly put it in the list of candidates
                if not verb_synset.isdisjoint(constraints_verbs['cause']):
                    candidates.append(
                        deepcopy([subtree.pos(), sibling.pos(), 'NP-VP-NP']))
                else:
                    # pattern contains a valid verb (that is not 'cause'), so check the 7 subpatterns
                    pre = [t[0]['nlpToken']['originalText'].lower() for t in subtree.pos() if
                           t[1][0] == 'N' and t[0]['nlpToken']['originalText'].isalpha()]
                    post = [t[0]['nlpToken']['originalText'].lower() for t in sibling.pos() if
                            t[1][0] == 'N' and t[0]['nlpToken']['originalText'].isalpha()]
                    pre_con = {'entity': False, 'abstraction': False}
                    post_con = {'entity': False, 'phenomenon': False, 'abstraction': False, 'group': False,
                                'possession': False, 'event': False, 'act': False, 'state': False}
                    verb_con = {'associate': False, 'relate': False,
                        'lead': False, 'induce': False}

                    # check nouns in after verb
                    for noun in post:
                        noun_synset = set(wordnet.synsets(noun, 'n'))
                        for con in post_con:
                            post_con[con] = post_con[con] or not noun_synset.isdisjoint(
                                constraints_hyponyms[con])
                            if post_con['phenomenon']:
                                break

                        if post_con['phenomenon']:
                            break

                    # check nouns in before verb
                    for noun in pre:
                        noun_synset = set(wordnet.synsets(noun, 'n'))
                        for con in pre_con:
                            pre_con[con] = pre_con[con] or not noun_synset.isdisjoint(
                                constraints_hyponyms[con])

                    # check if verb is relevant for a subpattern
                    for con in verb_con:
                        verb_con[con] = not verb_synset.isdisjoint(
                            constraints_verbs[con])
                        if verb_con[con]:
                            break

                    # apply subpatterns
                    if (
                            post_con['phenomenon']
                    ) or (
                            not pre_con['entity'] and (verb_con['associate'] or verb_con['relate']) and (
                            post_con['abstraction'] and post_con['group'] and post_con['possession'])
                    ) or (
                            not pre_con['entity'] and post_con['event']
                    ) or (
                            not pre_con['abstraction'] and (
                                post_con['event'] or post_con['act'])
                    ) or (
                            verb_con['lead'] and (
                                not post_con['entity'] and not post_con['group'])
                    ):
                        candidates.append(
                            deepcopy([subtree.pos(), sibling.pos(), 'NP-VP-NP']))

        for i in range(len(tokens)):
            token = tokens[i]['nlpToken']['originalText'].lower()

            if pos[i][1] == 'RB' and token in adverbial_indicators:
                # If we come along an adverb (RB) check the adverbials that indicate causation
                candidates.append(deepcopy([pos[:i], pos[i - 1:], 'RB']))

            elif token in causal_conjunctions and ' '.join(
                    [x['nlpToken']['originalText'] for x in tokens[i:]]).lower().startswith(
                    causal_conjunctions[token]):
                # Check if token is a clausal conjunction indicating causation
                start = i
                if token not in causal_conjunctions_inclusive:
                    # exclude clausal conjunction besides special cases
                    start += 1
                candidates.append(
                    deepcopy([pos[start:], pos[:i], 'biclausal']))

        # drop candidates containing other candidates
        unique_candidates = []
        candidate_strings = []
        for candidate in candidates:
            # Bugfix, at some very rare occasions, the candidate holds an empty list
            if len(candidate[0]) > 0:
                another_string = [x[0]['nlpToken']['originalText']
                    for x in candidate[1]]
                a_string = candidate[0][0][0]['nlpToken']['originalText'] + \
                    ' ' + ' '.join(another_string)
                candidate_strings.append(a_string)
                print(a_string,'\n')
        for i, candidate in enumerate(candidates):
            unique = True
            for j, substring in enumerate(candidate_strings):
                if i != j and candidate[2] == candidates[j][2] and substring in candidate_strings[i]:
                    unique = False
                    break
            if unique:
                unique_candidates.append(candidate)

        return unique_candidates


candidates = []
for i, tree in enumerate(postrees):
    for candidate in evaluate_tree(tree, adverbial_indicators, causal_conjunctions, causal_conjunctions_inclusive, causal_verbs, causal_verb_phrases, constraints_verbs, constraints_hyponyms):
        candidateObject = Candidate()
        candidateObject.set_raw(candidate[0])  # candidate[0] contains the cause, candidate[1] the effect
        candidateObject.set_type(candidate[2])
        candidateObject.set_sentence_index(i)

        candidates.append(candidateObject)

    doc.set_candidates(doc.get_id(), candidates)        
'''
x = []
zeze = {}
token_data_array = []
for candidateObject in candidates:
    for each in candidateObject:
        if type(each) == list:
            counter_index = 1
            for x in each:
                token_data = x[0]
                pos = x[1]
                for key, value in token_data.items():
                    inside_token_data = value
                    word = inside_token_data['word']
                    pos = inside_token_data['pos']
                    each = []
                    each.append(word)
                    each.append(pos)
                    token_data_array.append(each)
hmm = {}
sentence = ''
for item in token_data_array:
    word = item[0]
    if word not in sentence:
        sentence = sentence + ' ' + str(word)
#print(sentence)
'''
'''
nlp = stanza.Pipeline(
    lang='en', processors='tokenize,pos,lemma,depparse,ner,sentiment,constituency')
new_doc2 = nlp(sentence)

post_con = {'entity': False, 'phenomenon': False, 'abstraction': False, 'group': False,'possession': False, 'event': False, 'act': False, 'state': False}
lookup = {}
for sentence in new_doc2.sentences:
  print('sentence',sentence.text)
  tree = nltk.ParentedTree.fromstring(str(sentence.constituency))
  for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP' and t.right_sibling() is not None):
      sibling = subtree.right_sibling()

      while sibling.label() == 'ADVP' and sibling.right_sibling() is not None:
          sibling = sibling.right_sibling()

      if sibling.label() == 'VP' and "('NP'" in sibling.__repr__():
          verbs = [t[0] for t in sibling.pos() if t[1][0] == 'V'][:3]
          for verb in verbs:
              print('verb',verb)
              post = [t[0].lower() for t in sibling.pos()]
              arr = []
              for noun in post:
                  print("noun", noun)
                  noun_synset = set(wordnet.synsets(noun, 'n'))
                  for con in post_con:
                      if not noun_synset.isdisjoint(constraints_hyponyms[con]):
                          arr.append(noun)
                  lookup[verb] = arr
              
for key,value in lookup.items():
    print(key,set(value),'\n')
              
'''
