import logging
from Giveme5W1H.extractor.document import Document
from Giveme5W1H.extractor.extractor import MasterExtractor
from Giveme5W1H.extractor.candidate import Candidate
from Giveme5W1H.extractor.document import Document
import re
import os
from nltk.tree import ParentedTree
from Giveme5W1H.extractor.candidate import Candidate
from itertools import zip_longest
import csv
import spacy
import nltk
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from copy import deepcopy


DM_SINGLE_CLOSE_QUOTE = "\u2019"  # unicode
DM_DOUBLE_CLOSE_QUOTE = "\u201d"
END_TOKENS = [".", "!", "?", "...", "'", "`", '"',
              DM_SINGLE_CLOSE_QUOTE, DM_DOUBLE_CLOSE_QUOTE, ")"]


def get_hyponyms(synsets):
    result = set()
    for hyponym in synsets.hyponyms():
        result |= get_hyponyms(hyponym)
    return result | set(synsets.hyponyms())


def read_text_file_path(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    return lines


def get_hash_from_path(p):
    """Extract hash from path."""
    return os.path.splitext(os.path.basename(p))[0]


def get_art_abs(story_file):
    lines = read_text_file_path(story_file)
    file_name = get_hash_from_path(story_file)

    def fix_missing_period(line):
        """Adds a period to a line that is missing a period."""
        if "@highlight" in line:
            return line
        if not line:
            return line
        if line[-1] in END_TOKENS:
            return line
        return line + " ."

    lines = [fix_missing_period(line) for line in lines]

    # Separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    for line in lines:
        if not line:
            continue  # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    # Make article into a single string
    article = " ".join(article_lines)

    abstract = highlights

    return article, abstract, file_name


text9 = '''Barack Obama was born in Hawaii.  He is the president.'''

article, abstract, file_name = get_art_abs(
    "/Users/cham/Desktop/alex_event_data/attachments/extradite.txt")

date_publish = '2016-11-10 07:44:00'

log = logging.getLogger('GiveMe5W')
log.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
log.addHandler(sh)

# giveme5w setup - with defaults
extractor = MasterExtractor()
doc = Document.from_text(article, date_publish)
doc = extractor.parse(doc)
doc_len = doc.get_len()
corefs = doc.get_corefs()
trees = doc.get_trees()
candidates = []

weights = [0.9, 0.095, 0.005]
minimal_length_of_tokens = 3


def filter_candidate_dublicates(candidates):

    unique_map = {}
    unique_candidates = []
    for candidate in candidates:
        text = candidate.get_parts_as_text()
        text_id = ''.join(e for e in text if e.isalnum())
        if unique_map.get(text_id):
            # skip this one
            continue
        else:
            unique_candidates.append(candidate)
            unique_map[text_id] = True

    return unique_candidates


def cut_what(tree, min_length=0, length=0):

    if type(tree[0]) is not ParentedTree:
        # we found a leaf
        return ParentedTree(tree.label(), [tree[0]])
    else:
        children = []
        for sub in tree:
            child = cut_what(sub, min_length, length)
            length += len(child.leaves())
            children.append(child)
            if sub.label() == 'NP':
                sibling = sub.right_sibling()
                if length < min_length and sibling is not None and sibling.label() == 'PP':
                    children.append(sibling.copy(deep=True))
                break
        return ParentedTree(tree.label(), children)


def filter_duplicates(candidates, exact=True):

    mentioned = []
    filtered = []

    for candidate in candidates:

        string_a = []
        for part in candidate[0]:
            string_a.append(part[0]['nlpToken']['lemma'])
        string = ' '.join(string_a)

        if exact:
            new = string not in mentioned
        else:
            for member in mentioned:
                if string in member:
                    new = False
                    break

        mentioned.append(string)

        cd = Candidate()
        cd.set_parts(candidate[0])
        cd.set_score(candidate[1])
        cd.set_sentence_index(candidate[2] if 2 < len(candidate) else None)
        cd.set_type(candidate[3] if 3 < len(candidate) else None)

        filtered.append(cd)

    return filtered


def filterAndConvertToObjectOrientedList(list):
    max = 0
    candidates = filter_duplicates(list)
    for candidate in candidates:
        if candidate.get_score() > max:
            max = candidate.get_score()

    for candidate in candidates:
        score = candidate.get_score()
        candidate.set_score(score / max)

    # sort
    candidates.sort(key=lambda x: x.get_score(), reverse=True)

    return candidates


def evaluate_tree(sentence_root):

    candidates = []
    for subtree in sentence_root.subtrees():
        if subtree.label() == 'NP' and subtree.parent().label() == 'S':

            # Skip NPs containing a VP
            if any(list(subtree.subtrees(filter=lambda t: t.label() == 'VP'))):
                continue

            # check siblings for VP
            sibling = subtree.right_sibling()
            while sibling is not None:
                if sibling.label() == 'VP':

                    entry = [subtree.pos(), cut_what(sibling, minimal_length_of_tokens).pos(),
                             sentence_root.stanfordCoreNLPResult['index']]
                    candidates.append(entry)
                    break
                sibling = sibling.right_sibling()
    return candidates

#corefs = doc.get_corefs()


for cluster in corefs:
    print("cluster:", cluster)
    print("corefs[cluster]:", corefs[cluster])
    for mention in corefs[cluster]:
        print("sentence_number:", mention['sentNum'])
        for pattern in evaluate_tree(trees[mention['sentNum'] - 1]):
            np_string = ' '.join([p[0]['nlpToken']['originalText']
                                  for p in pattern[0]])
            print("pattern[0]", np_string, '\n')
            if re.sub(r'\s+', '', mention['text']) in np_string:
                candidate_object = Candidate()
                another_one = ' '.join([p[0]['nlpToken']['originalText']
                                       for p in pattern[1]])
                print('pattern[1]', another_one, '\n')
                print(
                    "==================================================================")
                '''print("SETTING WHATS IN THE DOCUMENT")
                print("pattern[0]: ", pattern[0])
                print("pattern[1]: ", pattern[1])
                print("cluster: ", cluster, '\n')
                print("mention id: ", mention["id"], '\n')
                print(
                    "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
'''
                candidate_object.set_sentence_index(pattern[2])
                candidate_object.set_raw(
                    [pattern[0], pattern[1], cluster, mention['id']])
                candidates.append(candidate_object)


doc.set_candidates('what', candidates)

ranked_candidates = []
doc_len = doc.get_len()
doc_ner = doc.get_ner()
doc_coref = doc.get_corefs()
postrees = doc.get_trees()


if any(doc_coref.values()):
    # get length of longest coref chain for normalization
    max_len = len(max(doc_coref.values(), key=len))
else:
    max_len = 1

for candidate in doc.get_candidates("what"):
    candidateParts = candidate.get_raw()
    #print("candidateParts:", candidateParts)
    # for item in candidateParts:
    #  print(item, 'n')
    # print("oooooooooooooooooooooooo")
    verb = candidateParts[1][0][0]['nlpToken']['originalText'].lower()
    # print(verb)
    # VP beginning with say/said often contain no relevant action and are therefor skipped.
    if verb.startswith('say') or verb.startswith('said'):
        continue

    coref_chain = doc_coref[candidateParts[2]]

    #print("cluster", candidateParts[2])
    #print("mention[id]", candidateParts[3])
    #print("coref_chain", coref_chain)
    # print("======================================================================================")

    # print(coref_chain)

    # first parameter used for ranking is the number of mentions, we use the length of the coref chain
    score = (len(coref_chain) / max_len) * weights[1]

    representative = None
    contains_ne = False
    mention_type = ''

    for mention in coref_chain:
        if mention['id'] == candidateParts[3]:
            # print(mention)
            #print("mention that matched:", mention)
            #print("candidateParts[3]:", candidateParts[3])
            mention_type = mention['type']
            if mention['sentNum'] < doc_len:
                # The position (sentence number) is another important parameter for scoring.
                # This is inspired by the inverted pyramid.
                score += ((doc_len -
                          mention['sentNum'] + 1) / doc_len) * weights[0]
        if mention['isRepresentativeMention']:
            # The representative name for this chain has been found.
            tmp = doc._sentences[mention['sentNum'] -
                                 1]['tokens'][mention['headIndex'] - 1]
            representative = ((tmp['originalText'], tmp), tmp['pos'])
            # print(representative)
            #print("representative", representative)
            try:
                # these dose`t work, if some special characters are present
                if representative[-1][1] == 'POS':
                    representative = representative[:-1]
            except IndexError:
                pass

        if not contains_ne:
            # If the current mention doesn't contain a named entity, check the other members of the chain
            for token in doc_ner[mention['sentNum'] - 1][mention['headIndex'] - 1:mention['endIndex'] - 1]:
                #print("token:", token)
                if token[1] in ['PERSON', 'ORGANIZATION', 'LOCATION']:
                    contains_ne = True
                    break

    if contains_ne:
        # the last important parameter is the entailment of a named entity
        score += weights[2]

    if score > 0:
        # normalize the scoring
        score /= sum(weights)

    if mention_type == 'PRONOMINAL':
        # use representing mention if the agent is only a pronouns
        rp_format_fix = [
            (({'nlpToken': representative[0][1]}, representative[0][1]['pos']))]
        #print("rp_format_fix", rp_format_fix)
        ranked_candidates.append(
            (rp_format_fix, candidateParts[1], score, candidate.get_sentence_index()))
    else:
        ranked_candidates.append(
            (candidateParts[0], candidateParts[1], score, candidate.get_sentence_index()))
print("========================================================================================================")

# split results
who = [(c[0], c[2], c[3]) for c in ranked_candidates]

what = [(c[1], c[2], c[3]) for c in ranked_candidates]

who_what = [(c[0], c[1], c[2], c[3]) for c in ranked_candidates]


final_data = []

for c in ranked_candidates:
    who_what = {}
    who = c[0]
    who_names = ''
    what_names = ''
    what = c[1]
    score = c[2]
    for x in who:
        name_dict = x[0]
        for key, value in name_dict.items():
            name_data = value
            name = name_data['word']
            who_names = who_names + ' ' + name

    for x in what:
        what_dict = x[0]
        for key, value in what_dict.items():
            what_data = value
            name = what_data['word']
            what_names = what_names + ' ' + name

    who_what[who_names] = what_names
    final_data.append([who_what, score])

sorted_list2 = sorted(final_data, key=lambda x: x[1], reverse=True)


NLP = spacy.load("en_core_web_lg")


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

why_data = []
candidates.sort(key=lambda x: x.get_score(), reverse=True)
for item in candidates:
    text = item.get_raw()
    pos = {}
    new_pos = []
    why_data_inside = {}
    # print(text)
    sentence = ''
    for each in text:
        token_data = each[0]
        for key, value in token_data.items():
            inside_token_data = value
            word = inside_token_data['word']
            sentence = sentence + ' ' + word
    why_data_inside[len(sentence)] = sentence
    why_data.append([why_data_inside, item.get_score()])
    document = NLP(str(sentence))
    for token in document:
        pos[token] = token.pos_
    if len(pos) > 5:
        for key, value in pos.items():
            if value == "NOUN":
                new_pos.append([key, value])

    # attack
'''
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
                        '''

#print(item.get_score(), item.get_type(), set(final_candidates), '\n')
# print(new_pos)
# print("=====================================================================================================================")

doc.set_answer('why', candidates)

d = [sorted_list2, why_data]
export_data = zip_longest(*d, fillvalue='')


with open('/Users/cham/Desktop/alex_event_data/attachments/extradite.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(("Who_what_combined",
                    "why"))
    writer.writerows(export_data)
f.close()
