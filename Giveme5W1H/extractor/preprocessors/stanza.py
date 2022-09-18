from stanza.server import CoreNLPClient, StartServer
text = "Chris Manning is a nice person. Chris wrote a simple sentence. He also gives oranges to people."
CUSTOM_PROPS = {
            'timeout': 500000,
            'annotators': 'tokenize,ssplit,pos,lemma,parse,ner,depparse,mention,coref',
            'tokenize.language': 'English',
            'outputFormat': 'json'
        }

with CoreNLPClient(properties=CUSTOM_PROPS, output_format="json") as client:
    annotation = client.annotate(text=text)
    for sentence in annotation['sentences']:
        print(sentence['parse'])
    
