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
import stanza

from copy import deepcopy


lead = '''The death toll from a powerful Taliban truck bombing at the German consulate in Afghanistan's Mazar-i-Sharif city rose to at least six Friday, with more than 100 others wounded in a major militant assault. The Taliban said the bombing late Thursday, which tore a massive crater in the road and overturned cars, was a "revenge attack" for US air strikes this month in the volatile province of Kunduz that left 32 civilians dead. The explosion, followed by sporadic gunfire, reverberated across the usually tranquil northern city, smashing windows of nearby shops and leaving terrified local residents fleeing for cover. The suicide attacker rammed his explosives-laden car into the wall of the German consulate, local police chief Sayed Kamal Sadat told AFP. All German staff from the consulate were unharmed, according to the foreign ministry in Berlin.'''

text1 = '''Severe storms that brought damaging winds, heavy rains and flash flooding to parts of the Midwest and the South were blamed for the deaths of three people, including two children in Michigan and Arkansas as well as a woman in Ohio.Monday's storms also knocked out electrical service to hundreds of thousands of homes and businesses in Indiana and Michigan, with dozens of schools canceling classes in Michigan alone on Tuesday because of power outages.In the Michigan city of Monroe, a 14-year-old girl was electrocuted Monday night in the backyard of her home after coming into contact with an electrical line that was knocked down by a thunderstorm, the public safety department said in a Facebook post.'''
date_publish = '2016-11-10 07:44:00'

text2 = '''Social media giant Snapchat has revealed it is sacking 1300 staff – which amounts to 20 per cent of its workforce globally – as its hit by an advertising downturn.
Its latest quarterly revenue growth of 8 per cent was “well below” expectations and the company’s cuts were part of a worst-case scenario plan where it would continue to be impacted next year by the weak advertising market, it said.'''

text3 = '''-LRB- CNN -RRB- -- On Tuesday , an Afghan soldier killed a U.S. major general and wounded a German brigadier general , as well as up to 15 others , in an attack at the Marshal Fahim National Defense University in the Afghan capital , Kabul . The attack is an ominous sign regarding the potential risks to American service members as the majority of U.S. forces withdraw from Afghanistan .
If a Bilateral Security Agreement between the United States and Afghanistan is signed in coming months , the United States is likely to keep a residual force of around 9,800 troops in Afghanistan after the withdrawal of all U.S. combat troops at the end of 2014 .This residual force would serve in an advisory role to Afghan troops , which could further expose American forces to insider attacks .
Officials identified the American who was killed as Maj. Gen. Harold Greene . Rear Adm. John Kirby , the Pentagon spokesman , said the Afghan soldier who carried out the attack was shot and killed .
Although directed at high-ranking officers , it was far from the first insider attack on coalition troops in Afghanistan .'''

text4 = '''June 28 (Reuters) - U.S. oil major Exxon Mobil Corp (XOM.N) and Imperial Oil Ltd (IMO.TO) said on Tuesday they will sell their Montney and Duvernay shale oil and gas assets in Canada to Whitecap Resources Inc (WCP.TO) for C$1.9 billion ($1.48 billion).

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
        result = set()
        for hyponym in synsets.hyponyms():
            result |= get_hyponyms(hyponym)
        return result | set(synsets.hyponyms())


# giveme5w setup - with defaults
extractor = MasterExtractor()
doc = Document.from_text(lead, date_publish)
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
constraints_verbs = {'cause': None, 'associate': None,'relate': None, 'lead': None, 'induce': None,'entity': None, 'phenomenon': None, 'abstraction': None, 'group': None, 'possession': None,
                            'event': None, 'action': None, 'state': None}

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
        candidates = []

        # Searching for cause-effect relations that involve a verb/action we look for NP-VP-NP
        for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP' and t.right_sibling() is not None):
            sibling = subtree.right_sibling()

            # skip to the first verb
            while sibling.label() == 'ADVP' and sibling.right_sibling() is not None:
                sibling = sibling.right_sibling()

            # NP-VP-NP pattern found .__repr__()
            if sibling.label() == 'VP' and "('NP'" in sibling.__repr__():
                verbs = [t[0] for t in sibling.pos() if t[1][0] == 'V'][:3]
                for verb in verbs:
                    normalized = verb['nlpToken']['originalText'].lower()
                    candidates.append(normalized)
                    
                    verb_synset = set(wordnet.synsets(normalized, 'v'))
                    post = [t[0]['nlpToken']['originalText'].lower() for t in sibling.pos() if t[1][0] == 'N' and t[0]['nlpToken']['originalText'].isalpha()]                    
                    post_con = {'entity': False, 'phenomenon': False, 'abstraction': False, 'group': False,'possession': False, 'event': False, 'act': False, 'state': False}
                    arr = []
                    for item in constraints_verbs:
                        if not verb_synset.isdisjoint(constraints_verbs[item]):
                            candidates.append(normalized)

       
        #for key,value in candidates.items():
            #print(key,value,'\n')
        return candidates


f=[]
for i, tree in enumerate(postrees):
    for candidate in evaluate_tree(tree, adverbial_indicators, causal_conjunctions, causal_conjunctions_inclusive, causal_verbs, causal_verb_phrases, constraints_verbs, constraints_hyponyms):
        f.append(candidate)
        
verbs = list(set(f))
ind_sentences = []
NLP = spacy.load("en_core_web_lg")
doc = NLP(lead)
for sentence in doc.sents:
    ind_sentences.append(str(sentence))

candidate_sents = []
for verb in verbs:
    for sent in ind_sentences:
        if verb in sent:
            candidate_sents.append(sent)
            
for item in list(set(candidate_sents)):
    print(item,'\n')
            