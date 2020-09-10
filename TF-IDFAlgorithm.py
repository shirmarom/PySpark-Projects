# Imports: enviroment preparations

import math 
from pyspark.sql.types import ArrayType, StructField, StructType, StringType, IntegerType
from pyspark.sql.functions import desc, row_number, monotonically_increasing_id
from pyspark.sql.window import Window

from pyspark.sql import SparkSession  
scSpark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example: Reading CSV file without mentioning schema") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
    
# Data preparations

def stringCleanex(x):
  x = x.lower()
  newString = ""
  for i in x:
    if ord(i) == 32 or (ord(i) > 96 and ord(i) < 123):
      newString += i
  newString = " ".join(newString.split())
  newLine = []
  for j in newString.split(" "):
    if j not in ['our', 'about', 'the', 'are', 'with', 'for', 'out', 'self', 'isnt', 'arent', 'http', 'ourselves', 'myself', 'yourself', 'too'] and len(j) > 2 and 'http' not in j:
      newLine.append(j)
  return " ".join(newLine)

def data_prep(path):
  firstCorpus = scSpark.read.csv(path, header=True, sep=",")
  firstCorpus.registerTempTable('firstCorpus')
  # Taking the second column only out of the data
  onlyText = scSpark.sql("""
  select text
  from firstCorpus
  where text != 'text' and text != ' '
  """)
  onlyTextRdd = onlyText.rdd.flatMap(list)
  # Cleaning the data, including special characters and stop words
  corpus = onlyTextRdd.map(stringCleanex)
  return corpus

def data_prep_tf(path):
  # Adding row numbers to the data, because tf-idf values are calculated per document
  corpus = data_prep(path)
  corpusNew = sqlContext.createDataFrame(corpus, StringType()) 
  corpusNew.registerTempTable('corpusNew')
  corpusIds = scSpark.sql("""
  select value, ROW_NUMBER() OVER(ORDER BY value) as Id
  from corpusNew
  """)
  return corpusIds # Returns a list of all lines with ids

def split_words(path):
  corpusIds = data_prep_tf(path) # List of all lines with ids
  rdd2 = corpusIds.rdd.map(lambda x: [(i, x[1]) for i in x[0].split(" ")]) # Splitting each line per words
  return rdd2

def words_for_tf(path):
  rdd2 = split_words(path)
  corpusWords = rdd2.flatMap(lambda x: x) # Merged list of all words and their relevant origin document number
  corpusTf = corpusWords.map(lambda x: (x, 1)).reduceByKey(lambda v1, v2: v1 + v2) # Number of words that appear in each document, per a pair of (word, document)
  return corpusTf

def words_per_line(path):
  corpusIds = data_prep_tf(path) # List of all lines - x[0] with their ids - x[1]
  all_words_per_line = corpusIds.rdd.map(lambda x: (x[1], len(x[0].split(" ")))).reduceByKey(lambda v1, v2: v1) # Calculating the number of words in each document
  dictRows = all_words_per_line.collectAsMap() # Creating a dictionary out of it, for tf calculations purposes
  return dictRows
  
 # Functions

def tfFunc(x, y):
  if y <= 1:
    return 0
  return ((math.log(x, 10) + 1) / math.log(y, 10))

def tfCalculator(path):
  corpusTf = words_for_tf(path) # List of all words and their counters per document
  dictRows = words_per_line(path) # Dictionary per line with number of words in line
  Tfs = corpusTf.map(lambda x: (x[0], tfFunc(x[1], dictRows[x[0][1]]))) # Calculating tf per (word, document)
  return Tfs

def idf_calc(path):
  corpus = data_prep(path)
  numberOfDocuments = corpus.count() # Number of documents
  wordsPerLine = corpus.flatMap(lambda x: list(dict.fromkeys(x.split(" "))))
  uniqueWordsPerLine = wordsPerLine.map(lambda x: (x, 1)).reduceByKey(lambda v1, v2: v1 + v2)
  idfs = uniqueWordsPerLine.map(lambda x: (x[0], math.log(1 + (numberOfDocuments / x[1]), 10)))
  return idfs # Returns all idf for all of the words in the corpus

def tf_idf(path):
  Tfs = tfCalculator(path)
  all_idfs = idf_calc(path)
  all_idfs_dict = all_idfs.collectAsMap()
  tfidfs = Tfs.map(lambda x: (x[0], x[1] * all_idfs_dict[x[0][0]]))
  return tfidfs
  
 
 
