from asyncio import set_event_loop
import logging
from readline import write_history_file
import nltk
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from Giveme5W1H.extractor.document import Document
from Giveme5W1H.extractor.extractor import MasterExtractor
from Giveme5W1H.extractor.candidate import Candidate
from typing import List
from Giveme5W1H.extractor.combined_scoring.abs_combined_scoring import AbsCombinedScoring
from Giveme5W1H.extractor.document import Document
from copy import deepcopy
import spacy
from pprint import pprint
from operator import itemgetter
from nltk.corpus import framenet as fn
from nltk.corpus.reader.framenet import PrettyList
# airstrikes
# air strikes

text1 = '''The death toll from a powerful Taliban truck bombing at the German consulate in Afghanistan's Mazar-i-Sharif city rose to at least six Friday, with more than 100 others wounded in a major militant assault. The Taliban said the bombing late Thursday, which tore a massive crater in the road and overturned cars, was a "revenge attack" for US airstrikes this month in the volatile province of Kunduz that left 32 civilians dead. The explosion, followed by sporadic gunfire, reverberated across the usually tranquil northern city, smashing windows of nearby shops and leaving terrified local residents fleeing for cover. The suicide attacker rammed his explosives-laden car into the wall of the German consulate, local police chief Sayed Kamal Sadat told AFP. All German staff from the consulate were unharmed, according to the foreign ministry in Berlin.'''

date_publish = '2016-11-10 07:44:00'

text2 = '''-LRB- CNN -RRB- -- On Tuesday , an Afghan soldier killed a U.S. major general and wounded a German brigadier general , as well as up to 15 others , in an attack at the Marshal Fahim National Defense University in the Afghan capital , Kabul . The attack is an ominous sign regarding the potential risks to American service members as the majority of U.S. forces withdraw from Afghanistan .
If a Bilateral Security Agreement between the United States and Afghanistan is signed in coming months , the United States is likely to keep a residual force of around 9,800 troops in Afghanistan after the withdrawal of all U.S. combat troops at the end of 2014 .This residual force would serve in an advisory role to Afghan troops , which could further expose American forces to insider attacks .
Officials identified the American who was killed as Maj. Gen. Harold Greene . Rear Adm. John Kirby , the Pentagon spokesman , said the Afghan soldier who carried out the attack was shot and killed .
Although directed at high-ranking officers , it was far from the first insider attack on coalition troops in Afghanistan .'''

text3 = '''June 28 U.S. oil major Exxon Mobil Corp and Imperial Oil Ltd said on Tuesday they will sell their Montney and Duvernay shale oil and gas assets in Canada to Whitecap Resources Inc (WCP.TO) for C$1.9 billion ($1.48 billion). Exxon and Imperial, which jointly own the assets, began marketing them at the start of this year, hoping to capitalize on a rebound in oil and gas prices.The assets were valued at up to $1 billion in January by industry insiders. A strong run-up in commodity prices since then, with Russia's invasion of Ukraine stoking global supply concerns, has pushed up the value of oil and gas properties across North America. U.S. crude oil futures settled at $111.76 a barrel on Tuesday, up about 49% so far this year. Imperial's share in the sale will be around C$940 million, the companies said on Tuesday.
The assets being sold include 567,000 net acres in the Montney shale play, 72,000 net acres in the Duvernay basin and additional acreage in other areas of Alberta. Net production from the assets is about 140 million cubic feet of natural gas per day and about 9,000 barrels of crude, condensate and natural gas liquids per day, according to the companies. The shale assets were related to a multi-billion dollar impairment charge that Imperial and Exxon took in late 2020. The companies also own petrochemical plants and Exxon operates offshore production in Eastern Canada. The asset sale is part of Exxon's plans to divest smaller oil and gas operations as it looks to pay down debt and reward shareholders. For Imperial, it is part of its "strategy to focus upstream resources on key oil sands assets", the company said.
The sale is expected to close before the end of the third quarter.
'''

text4 = '''Sept 4 Japan's Toyota Motor Corp's truck and bus unit Hino Motors will halt production of some medium and heavy-duty trucks for at least another year after a widespread data falsification scandal, Nikkei Asia reported on Sunday. The medium-duty Ranger and the heavy-duty Profia truck will not be produced until August 2023, the report added. Halting production of some truck models is the latest sign of the scandal worsening for Hino since it first announced the data falsification affecting some of its bigger trucks in March. Since then, it has said it falsified data on some engines going back as far as 2003, at least a decade earlier than originally indicated. All told, about 640,000 vehicles have been affected, or more than five times the figure initially revealed. Hino said last month it would suspend shipments of small trucks after a transport ministry investigation revealed that some 76,000 of its small trucks sold since 2019 had not been subject to the required number of engine tests. Toyota and others involved in a commercial vehicle partnership have since expelled Hino from the group over falsification of engine data by the truckmaker. The widening scandal at Japan's Hino Motors over falsification of engine data has become a headache that will not go away for parent Toyota which has a controlling 50.1% stake in Hino. Hino became Toyota's subsidiary in 2001 and nearly all Hino presidents since then previously worked for Toyota.Toyota did not immediately respond to a request for comment, and Hino could not immediately be reached.
'''

text5 = '''Aug 2 - Canada's Toronto Dominion Bank (TD.TO) said it will buy New York-based boutique investment bank Cowen Inc (COWN.O) for $1.3 billion in cash, seeking to boost its presence in the high-growth U.S market.The deal marks TD's second acquisition bid in the United States this year and the Canada's second-largest lender by market value has made no secret of its ambitions to expand in the world's biggest economy. TD will fund the acquisition from the $1.9 billion proceeds from the sale of shares of Charles Schwab , announced on Monday.
Canada's major banks have been on a shopping spree south of the border in the past year, seeking growth away from home, where the Big Six banks control nearly 90% of the market. read more
National Bank Financial analysts said the deal provides "valuable diversification" of TD's U.S. capital markets business, but flagged integration as a primary risk, saying it is "notoriously difficult when involving investment banking operations with different cultures."
In February, TD said it would buy Memphis-based First Horizon Corp (FHN.N) for $13.4 billion in its biggest ever acquisition. read more 
Investors had already expressed concern around integration following the First Horizon deal. read more
The Cowen deal values the target at $39 a share, a nearly 10% premium to its last closing price. Cowen shares rose 8.9% in morning trading in New York. TD shares slipped 1.3%.
The transaction is "modestly positive (for TD), especially given that it is on-strategy with the bank’s push into its U.S.-dollar platform," Credit Suisse Analyst Joo Ho Kim wrote in a note.
On Monday, TD said it was selling 28.4 million shares of Schwab, reducing its ownership to about 12% from 13.4%. That stake was the result of Schwab's purchase of TD Ameritrade, of which TD owned 43%. TD said it has no current plans to sell more Schwab shares.
TD expects pre-tax integration costs of about $450 million over three years, and revenue synergies of $300-350 million by the third year. TD must pay a termination fee of $42.25 million if it cancels the deal because of a recommendation change or another superior proposal.
The deal is expected to close in the first quarter of 2023.'''

text6 = '''Susan Bannigan has been appointed as the new Board Chair of Milk Crate Theatre.

Bannigan has joined Milk Crate Theatre having recently moved on from her position as CEO of the Westpac Foundation and Westpac Scholarship Trust where she worked closely with Westpac, community groups, social entrepreneurs and the business sector to support new innovations in addressing the complex issues of homelessness, long-term unemployment, social inclusion for refugees and those living with issues of mental health in communities across Australia.

She has over 30 years’ experience in experience in the financial services and philanthropic industries in Europe, Pacific and Australia. Bannigan’s former Board roles include Chair of the Business/Higher Education Round Table and Director of Variety NSW. She is a Chartered Accountant, member of the Australian Institute of Company Directors and holds a Bachelor’s degree in Economics from the University of Sydney.

Milk Crate Theatre CEO, Jodie Wainwright, said: ‘She brings extensive leadership and governance experience from her distinguished career in banking and with the Westpac Foundation. Susan has been a long-term supporter and her skills will be of immense value to Milk Crate Theatre. The board and team look forward to working with Susan to realise our vision of effecting social change through the power of performance,’ Wainwright said.

Bannigan will replace Michael Sirmai, who has been a member of the Milk Crate Board since 2013 and Board Chair since 2016.

‘Under Michael’s stewardship we have been able to navigate many of the challenges that faced small to medium arts organisations and come through the COVID pandemic in a strong position, poised for growth,’ said Wainwright. 

‘During his period as Chair, we have created a new and sustainable business plan and structure. We have also invested in our Social Impact Framework, with a robust Theory of Change and have embedded impact measurement into the program design. We thank Michael for his leadership and wish him well in his future endeavours,’ she added.

Michael Sirmai added: ‘I am absolutely delighted for someone of Susan’s experience and reputation to lead the Board over the company’s next phase.’

Bannigan commenced her role as Board Chair effective 15 August.'''

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


NLP = spacy.load("en_core_web_lg")

# giveme5w setup - with defaults
extractor = MasterExtractor()
doc = Document.from_text(text6, date_publish)
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
    for i, candidate in enumerate(candidates):
        unique = True
        for j, substring in enumerate(candidate_strings):
            if i != j and candidate[2] == candidates[j][2] and substring in candidate_strings[i]:
                unique = False
                break
        if unique:
            unique_candidates.append(candidate)
    items = []
    for item in unique_candidates:
        sentence = ''
        for each in item:
            if type(each) == list:
                for x in each:
                    token_data = x[0]
                    pos = x[1]
                    for key, value in token_data.items():
                        inside_token_data = value
                        word = inside_token_data['word']
                        sentence = sentence + ' ' + word
        items.append(sentence)

        # for item in items:
        # print(item,'\n')
    return unique_candidates


candidates = []

for i, tree in enumerate(postrees):
    for candidate in evaluate_tree(tree, adverbial_indicators, causal_conjunctions, causal_conjunctions_inclusive, causal_verbs, causal_verb_phrases, constraints_verbs, constraints_hyponyms):
        candidateObject = Candidate()
        # print("candidate[1]",candidate[1],'\n',"candidate[2]",candidate[2],'\n')
        # candidate[0] contains the cause, candidate[1] the effect
        candidateObject.set_raw(candidate[1])
        candidateObject.set_type(candidate[2])
        candidateObject.set_sentence_index(i)

        candidates.append(candidateObject)

    doc.set_candidates("why", candidates)


candidates = doc.get_candidates("why")
weights = [.60, .50, .30, .04]
# normalization sum is only first and second weight, because the second to fourth weights
# are only virtual weights but actually scores
weights_norm_sum = weights[0] + weights[1]

for candidateObject in candidates:
    parts = candidateObject.get_raw()
    # print(type(parts),parts,'\n')
    if parts is not None and len(parts) > 0:
        score = 0
        if candidateObject.get_type() == 'biclausal':
            score += weights[1]
        elif candidateObject.get_type() == 'RB':
            score += weights[2]
        else:
            score += weights[3]

        if score > 0:
            score /= weights_norm_sum

            # NEW
        candidateObject.set_score(score)

        # TODO: remove leftover from refactoring
for candidate in candidates:
    candidate.set_parts(candidate.get_raw())

candidates.sort(key=lambda x: x.get_score(), reverse=True)
for item in candidates:
    text = item.get_raw()
    pos = {}
    new_pos = []
    # print(text)
    sentence = ''
    for each in text:
        token_data = each[0]
        for key, value in token_data.items():
            inside_token_data = value
            word = inside_token_data['word']
            sentence = sentence + ' ' + word

    document = NLP(str(sentence))
    for token in document:
        pos[token] = token.pos_
    if len(pos) > 5:
        for key, value in pos.items():
            if value == "NOUN":
                new_pos.append([key, value])

    # attack

    if new_pos:
        final_candidates = []
        for each in new_pos:
            word = each[0]
            frames = [f.name for f in fn.frames(str(word))]
            for frame in frames:
                f = fn.frame(frame)
                lex = f.lexUnit
                for each in lex:
                    lex_word = each[:-2]
                    if lex_word in sentence:
                        final_candidates.append(sentence)

        print(item.get_score(), item.get_type(), set(final_candidates), '\n')
        # print(new_pos)
    print("=====================================================================================================================")

doc.set_answer('why', candidates)
