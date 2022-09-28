import logging
from turtle import distance
from Giveme5W1H.extractor.document import Document
from Giveme5W1H.extractor.extractor import MasterExtractor
from Giveme5W1H.extractor.candidate import Candidate
from Giveme5W1H.extractor.document import Document
import re
import os
from nltk.tree import ParentedTree
from Giveme5W1H.extractor.candidate import Candidate
from copy import deepcopy
from nltk.translate.bleu_score import sentence_bleu


date_publish = '2016-11-10 07:44:00'

candidates = []

minimal_length_of_tokens = 4


def read_text_file_path(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    return lines


DM_SINGLE_CLOSE_QUOTE = "\u2019"  # unicode
DM_DOUBLE_CLOSE_QUOTE = "\u201d"
END_TOKENS = [".", "!", "?", "...", "'", "`", '"',
              DM_SINGLE_CLOSE_QUOTE, DM_DOUBLE_CLOSE_QUOTE, ")"]


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


article, abstract, file_name = get_art_abs(
    "/home/cham/Desktop/attachments/transfer-ownership.txt")

# giveme5w setup - with defaults
extractor = MasterExtractor()
doc = Document.from_text(article, date_publish)

doc = extractor.parse(doc)


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
                if sibling is not None:
                    children.append(sibling.copy(deep=True))
                break
        return ParentedTree(tree.label(), children)


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
                    # this gives a tuple to find the way from sentence to leaf
                    # tree_position = subtree.leaf_treeposition(0)
                    entry = [subtree.pos(), cut_what(sibling, minimal_length_of_tokens).pos(),
                             sentence_root.stanfordCoreNLPResult['index']]
                    '''
                    np_string1 = ' '.join([p[0]['nlpToken']['originalText'] for p in entry[0]])
                    np_string2 = ' '.join([p[0]['nlpToken']['originalText'] for p in entry[1]])

                    print(np_string1)
                    print(np_string2)
                    print("=========================================================================================================")
                    '''
                    candidates.append(entry)
                    break
                sibling = sibling.right_sibling()
    return candidates


ranked_candidates = []
doc_len = doc.get_len()
doc_ner = doc.get_ner()
doc_coref = doc.get_corefs()
postrees = doc.get_trees()
corefs = doc.get_corefs()
trees = doc.get_trees()


for cluster in corefs:
    for mention in corefs[cluster]:
        for pattern in evaluate_tree(trees[mention['sentNum'] - 1]):
            candidate_object = Candidate()
            candidate_object.set_sentence_index(pattern[2])
            candidate_object.set_raw(
                [pattern[0], pattern[1], cluster, mention['id']])

            candidates.append(candidate_object)

doc.set_candidates("what", candidates)
weights = [0.9, 0.095, 0.005]

if any(doc_coref.values()):
    # get length of longest coref chain for normalization
    max_len = len(max(doc_coref.values(), key=len))
else:
    max_len = 1

for candidate in doc.get_candidates("what"):
    candidateParts = candidate.get_raw()
    verb = candidateParts[1][0][0]['nlpToken']['originalText'].lower()
    if verb.startswith('say') or verb.startswith('said'):
        continue
    coref_chain = doc_coref[candidateParts[2]]
    score = (len(coref_chain) / max_len) * weights[0]

    representative = None
    contains_ne = False
    mention_type = ''

    for mention in coref_chain:
        if mention['id'] == candidateParts[3]:
            mention_type = mention['type']

        if mention['isRepresentativeMention']:
            tmp = doc._sentences[mention['sentNum'] -
                                 1]['tokens'][mention['headIndex'] - 1]
            representative = ((tmp['originalText'], tmp), tmp['pos'])

            try:
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
        score += weights[0]

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

# split results
who = [(c[0], c[2], c[3]) for c in ranked_candidates]

what = [(c[1], c[2], c[3]) for c in ranked_candidates]

who_what = [(c[0], c[1], c[2], c[3]) for c in ranked_candidates]

final_data = []

who_what_2 = []

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

    who_what_2.append(who_names + what_names)
    who_what[who_names] = what_names
    final_data.append([who_what, score])

who_data_3 = list(set(who_what_2))
sentence_scores = []
def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(set(list1)) + len(set(list2))) - intersection
    return float(intersection) / union

for item in who_data_3:
    score = jaccard_similarity(article.split(),item.split())
    sentence_scores.append([item,score])

sorted_scores = sorted(sentence_scores, key=lambda x: x[1], reverse=True)

for item in sorted_scores:
    print(item,'\n')

#for item in final_data:
    #print(item,'\n')
sorted_list2 = sorted(final_data, key=lambda x: x[1], reverse=True)



 

