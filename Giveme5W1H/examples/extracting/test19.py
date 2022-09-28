from Giveme5W1H.extractor.document import Document
from Giveme5W1H.extractor.extractor import MasterExtractor
from Giveme5W1H.extractor.document import Document
import os
from Giveme5W1H.extractor.candidate import Candidate
import pandas as pd
import numpy as np
from nltk.corpus import wordnet
from pyspark.ml import Pipeline
import pyspark.sql.functions as F
from sparknlp.annotator import *
from sparknlp.base import *
import sparknlp
from stanza.server import CoreNLPClient
from nltk import tokenize

def print_msg_box(msg, indent=1, width=None, title=None):
    """Print message-box with optional title."""
    lines = msg.split('\n')
    space = " " * indent
    if not width:
        width = max(map(len, lines))
    box = f'╔{"═" * (width + indent * 2)}╗\n'  # upper_border
    if title:
        box += f'║{space}{title:<{width}}{space}║\n'  # title
        box += f'║{space}{"-" * len(title):<{width}}{space}║\n'  # underscore
    box += ''.join([f'║{space}{line:<{width}}{space}║\n' for line in lines])
    box += f'╚{"═" * (width + indent * 2)}╝'  # lower_border
    print(box)

spark = sparknlp.start()



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


article, abstract, file_name = get_art_abs("/home/cham/Desktop/attachments/mock.txt")
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

ace_words = {'born': None, 'marry': None, 'divorce': None, 'injure': None, 'die': None,
                            'transport': None, 'transfer': None, 'ownership': None,'merge': None,'declare': None,
                            'bankruptcy': None,'attack': None,'demonstrate': None,'meet': None,'position': None,
                            'start': None,'nominate': None,'elect': None,'arrest': None,'jail': None,'release': None,
                            'parole': None,'trial': None,'charge': None,'indict': None,'sue': None,'convict': None,
                            'sentence': None,'fine': None,'execute': None,'extradite': None,'acquit': None,'pardon': None,
                            'appeal': None}

for verb in ace_words:
    ace_words[verb] = set(wordnet.synsets(verb, 'v'))


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

who_what_candidates = []
for item in postrees:
    output = evaluate_tree(item)
    who_what_candidates.append(output)


document = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

token = Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')


embedding = BertEmbeddings.pretrained(name="electra_large_uncased", lang='en') \
    .setInputCols(['document', 'token']) \
    .setOutputCol('embeddings')

ner = NerDLModel.pretrained("onto_electra_large_uncased", "en") \
    .setInputCols(['document', 'token', 'embeddings']) \
    .setOutputCol('ner')

ner_con = NerConverter() \
    .setInputCols(['document', 'token', 'ner']) \
    .setOutputCol('ner_chunk')

nlp_pipeline = Pipeline(stages=[
    document, 
    token,
    embedding,
    ner,
    ner_con
])


empty_df = spark.createDataFrame([['']]).toDF('text')
pipeline_model = nlp_pipeline.fit(empty_df)
df = spark.createDataFrame(pd.DataFrame({'text': who_what_candidates}))
result = pipeline_model.transform(df)
result.printSchema()
result_df = result.select('text',F.explode(F.arrays_zip(result.ner_chunk.result, result.ner_chunk.metadata)).alias("cols")) \
      .select('text',F.expr("cols['0']").alias("chunk"),
              F.expr("cols['1']['entity']").alias("ner_label"))

resultdf_grouped = result_df.groupby('text').agg(F.collect_set('chunk').alias('chunk_grouped'),F.collect_set('ner_label').alias('ner_labels_grouped')).toPandas()

final_output = {}

for index,row in resultdf_grouped.iterrows():
    text = row['text']
    chunk = row['chunk_grouped']
    ner = row['ner_labels_grouped']
    matched_data = {}
    wordnet_array = []
    for word in text.split():
        check_word_sentence = set(wordnet.synsets(word, 'v'))
        for each in ace_words:
            if not check_word_sentence.isdisjoint(ace_words[each]):
                wordnet_array.append(each)
                matched_data["ner_len"] = len(ner)
                matched_data["ner"] = ner
                matched_data["ner_chunk"] = chunk
    matched_data["matched_wordnet"] = wordnet_array
    final_output[text] = matched_data

###############################################################################################################################################################################################################################

print_msg_box('\n~ EVENTS ~\n')
for key,value in final_output.items():
    print(key,value,'\n') 
                
       


    




                   










 

