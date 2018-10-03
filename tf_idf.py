import csv
from gensim import corpora, models, similarities
from collections import defaultdict
from pprint import pprint
from six import iteritems
import re

stoplist = []
documents = []

with open("stopwords-en.csv", 'r') as csvfile:
	csvreader = csv.reader(csvfile)
	for row in csvreader:
		stoplist.append(row[0])

with open("ap.csv", 'r') as csvfile:
	csvreader = csv.reader(csvfile)
	for row in csvreader:
		documents.append(row[1])


documents =	[re.sub('\b[0-9][0-9.,-]*\b', 'NUMBER-TOKEN', line) for line in documents]

#Make the dictionary
dictionary = corpora.Dictionary(line.lower().split() for line in documents)
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
dictionary.filter_tokens(stop_ids + once_ids)
dictionary.compactify()

#Convert each document to a bag of words
class MyCorpus(object):
	def __iter__(self):
		for line in documents:
			yield dictionary.doc2bow(line.lower().split(' '))

corpus = MyCorpus()

tfidf = models.TfidfModel(corpus)
tfidf_corpus = tfidf[corpus]

model = models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=300)

model.print_topics(10,15)