import stanza
import spacy
import nltk

def read_text_file_path(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    return lines

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


def evaluate_tree(tree):
        # Searching for cause-effect relations that involve a verb/action we look for NP-VP-NP
        for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP' and t.right_sibling() is not None):
            sibling = subtree.right_sibling()

            # skip to the first verb
            while sibling.label() == 'ADVP' and sibling.right_sibling() is not None:
                sibling = sibling.right_sibling()

            # NP-VP-NP pattern found .__repr__()
            if sibling.label() == 'VP' and "('NP'" in sibling.__repr__():
              np_string1 = ' '.join([p[0] for p in subtree.pos()])
              np_string2 = ' '.join([p[0] for p in sibling.pos()])
              who_what = np_string1 +' '+ np_string2
              verbs = [t[0] for t in sibling.pos() if t[1][0] == 'V'][:3]
              return who_what,verbs
          
          
nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
sentence_splitter = spacy.load("en_core_web_md")
NER = spacy.load("en_core_web_md")

ner_desired = ['ORG','GPE','LOC','PER','DATE']
master_output = {}
index = 0
paragraphs = read_text_file_path("/Users/cham/Desktop/alex_event_data/21.txt")
for paragraph in paragraphs:
    if paragraph != '':
        print_msg_box(paragraph)
        index = index + 1
        print_msg_box("ParagraphID"+" "+str(index))
        individual_sentence = sentence_splitter(paragraph)
        print_msg_box("NER")
        print("==============================================")
        for sent in individual_sentence.sents:
            ners = NER(str(sent))
            for word in ners.ents:
                print(word.text,word.label_,'\n')
                
            doc = nlp(str(sent))
            for sent in doc.sentences:
                sentence_tree = nltk.ParentedTree.fromstring(str(sent.constituency))
                output,trigger_words = evaluate_tree(sentence_tree)
                print(output,'\n',trigger_words,'\n')
            
              
 
        
