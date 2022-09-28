from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from sparknlp.annotator import *
from sparknlp.base import *
import sparknlp
from sparknlp.pretrained import PretrainedPipeline
import pandas as pd
from stanza.server import CoreNLPClient
from nltk import tokenize


spark = sparknlp.start()

text = """The Special Investigations Section of the Tennessee Department of Revenue conducted the investigation that led to the indictment and arrest of Scharneitha Britton, owner, of Kinfolks BBQ, in Smyrna.  Revenue special agents arrested Britton, age 66, on Monday.  Bond was set at $25,000.

On September 8, the Rutherford County Grand Jury indicted Britton on one Class B felony charge of theft over $60,000, 13 counts of money laundering (also Class B) and 36 felony counts of tax evasion.  The indictments allege Britton underreported taxable sales and failed to remit additional sales tax collected from her customers.

“Investigations, such as this one, should warn retailers that failing to properly remit all the sales tax monies they collect is a crime, “ Revenue Commissioner David Gerregano said.  “The taxes collected from customers are property of the state and local governments at all times.  Customers have a right to know that the tax they pay will be remitted to the state and used for the public good of all Tennesseans.”

If convicted, Britton could be sentenced to a maximum of two years in the state penitentiary and fined up to $3,000 for each count of tax evasion, and 12 years and fined up to $25,000 for money laundering and theft.

The Department is pursuing this criminal case in cooperation with District Attorney Jennings Jones’ office.  Citizens who suspect violations of Tennessee's revenue laws should call the toll-free tax fraud hot line at (800) FRAUDTX (372-8389).

The Department of Revenue is responsible for the administration of state tax laws and motor vehicle title and registration laws and the collection of taxes and fees associated with those laws. The department collects about 87 percent of total state revenue.  During the 2021 fiscal year, it collected $18.4 billion in state taxes and fees and more than $3.7 billion in taxes and fees for local governments. """

documentAssembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')
    
sentence = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx") \
    .setInputCols("document") \
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(['sentence']) \
    .setOutputCol('token')


embeddings = BertEmbeddings.pretrained(name="electra_large_uncased", lang='en') \
    .setInputCols(['sentence', 'token']) \
    .setOutputCol('embeddings')

ner_model = NerDLModel.pretrained("onto_electra_large_uncased", "en") \
    .setInputCols(['sentence', 'token', 'embeddings']) \
    .setOutputCol('ner')

ner_converter = NerConverter() \
    .setInputCols(['sentence', 'token', 'ner']) \
    .setOutputCol('ner_chunk')

nlp_pipeline = Pipeline(stages=[
    documentAssembler,
    sentence, 
    tokenizer,
    embeddings,
    ner_model,
    ner_converter
])


empty_df = spark.createDataFrame([['']]).toDF('text')
pipeline_model = nlp_pipeline.fit(empty_df)
df = spark.createDataFrame([[text]]).toDF("text")
result = pipeline_model.transform(df)
result.printSchema()
result_df = result.select(F.explode(F.arrays_zip(result.ner_chunk.result, result.ner_chunk.metadata)).alias("cols")) \
      .select(F.expr("cols['0']").alias("chunk"),
              F.expr("cols['1']['entity']").alias("ner_label"))

print("Named entity recognition")
result_df.show(200,False)

result_df2 = result.select('text',F.explode(F.arrays_zip(result.ner_chunk.result, result.ner_chunk.metadata)).alias("cols")) \
      .select('text',F.expr("cols['0']").alias("chunk"),
              F.expr("cols['1']['entity']").alias("ner_label"))

resultdf_grouped2 = result_df2.groupby('text').agg(F.collect_set('chunk').alias('chunk_grouped'),F.collect_set('ner_label').alias('ner_labels_grouped')).toPandas()

ner_chunks = []
ner_each_items = []
for index,row in resultdf_grouped2.iterrows():
    ner_chunk = row['chunk_grouped']
    ner_chunks.append(ner_chunk)


for item in ner_chunks:
    for each in item:
        ner_each_items.append(each)



print("Relationship extraction")
with CoreNLPClient(annotators=['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref','openie'],memory='8G', be_quiet=False,endpoint = "http://localhost:9003") as client:
    ann = client.annotate(text)
    modified_text = tokenize.sent_tokenize(text)
    for coref in ann.corefChain:
        antecedent = []
        for mention in coref.mention:
            phrase = []
            for i in range(mention.beginIndex, mention.endIndex):
                phrase.append(ann.sentence[mention.sentenceIndex].token[i].word)
            if antecedent == []:
                antecedent = ' '.join(word for word in phrase)
            else:
                anaphor = ' '.join(word for word in phrase)
                modified_text[mention.sentenceIndex] = modified_text[mention.sentenceIndex].replace(anaphor, antecedent)
    modified_text = ' '.join(modified_text)

    ann2 = client.annotate(modified_text)
    #print(ann)
    for sentence in ann2.sentence:
        for triple in sentence.openieTriple:
            subject = str(triple.subject)
            relation = str(triple.relation)
            object = str(triple.object)
            if subject in ner_each_items and object in ner_each_items:
                print("relation_triplet:",triple.subject + " " +triple.relation+ " "+triple.object,'\n')
