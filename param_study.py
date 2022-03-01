# Various preprocessing techniques and n_topics for LDA to optimize Similarity score
# Author: Florian Lietz
# Last edited: 2022-02-21
# 
#%%
import os, sys, re
from nltk.chunk import ne_chunk
import pandas as pd
import numpy as np
from os import path
from argparse import ArgumentParser
from stages.utils.utils import parseArgs, DataCleaner
from stages.TM.textmining import Message, Case, CasesList, lemmatize, gen_words

#%%
# Import, preprocessing, case clustering
infile = path.join("resources", "dataset", "Mail_ApplicationDataset_-2.csv")
# import NLD file
inputFile = pd.read_csv(infile, delimiter=";")
cleaner = DataCleaner(
    removeURLs=True,
    removeMultWhitespace=True,
    lowercasing=False,
    dateFormat="%Y-%m-%d %H:%M:%S"
)
cleaner.apply(inputFile)
casesList = CasesList.groupCases(file=inputFile, maxDays=14)
#casesList.prettyprint()

#%%
# Text mining
# 1. step: Preprocesing

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
# gensim
import gensim
import gensim.corpora
from gensim.utils import pickle, simple_preprocess
from gensim.models import CoherenceModel
# spacy
import spacy

# vis
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

stopwords = stopwords.words("english")
corpora_msg = casesList.getCorpora()
len(corpora_msg)
#%%
corpora_lemmatized = lemmatize(corpora_msg)
corpora_prep = gen_words(corpora_lemmatized)

#%%
# 2. step: build up clusters using LDA and visualize them
# identify bigrams and trigrams
bigrams_phrases = gensim.models.Phrases(
    corpora_prep,
    min_count=5,
    threshold=20
)
trigram_phrases = gensim.models.Phrases(
    bigrams_phrases[corpora_prep],
    threshold=20
)
bigram = gensim.models.phrases.Phraser(bigrams_phrases)
trigram = gensim.models.phrases.Phraser(trigram_phrases)

def make_bigrams(texts):
    return([bigram[doc] for doc in texts])
def make_trigrams(texts):
    return ([trigram[bigram[doc]] for doc in texts])

data_bigrams = make_bigrams(corpora_prep)
data_bigrams_trigrams = make_trigrams(data_bigrams)
# Train with dataset
id2word = gensim.corpora.Dictionary(data_bigrams_trigrams)
corp = [id2word.doc2bow(text) for text in data_bigrams_trigrams]

#%% TF IDF removal - better not!

id2word = gensim.corpora.Dictionary(corpora_prep)
corp = [id2word.doc2bow(text) for text in corpora_prep]

from gensim.models import TfidfModel
tfidf = TfidfModel(corp, id2word=id2word)
low_value = 0.02 # threshold
words = []
words_missing_in_tfidf = []

for i in range(0,len(corp)):
    bow = corp[i]
    low_value_words = []
    tfidf_ids = [id for id,value in tfidf[bow]]
    bow_ids = [id for id,value in bow]
    low_value_words = [id for id,value in tfidf[bow] if value < low_value]
    drops = low_value_words+words_missing_in_tfidf
    for item in drops:
        words.append(id2word[item])
    words_missing_in_tfidf = [id for id in bow_ids if id not in tfidf_ids] # words with tfidf score == 0
    new_bow = [b for b in bow if b[0] not in low_value_words and b[0] not in words_missing_in_tfidf]
    corp[i] = new_bow

#%%
#%% LDA Model
id2word = gensim.corpora.Dictionary(corpora_prep)
corp = [id2word.doc2bow(text) for text in corpora_prep]

#%%
# Hyperparameter Optimization
# Start of optimization operation

num_pool = list(range(5,31))
pool_scores = []
for pool in num_pool:
    lda_model = gensim.models.ldamodel.LdaModel(
        corpus=corp, # corp
        id2word=id2word,
        num_topics=pool,
        random_state=100,
        update_every=1,
        chunksize=100,
        passes=10,
        alpha="auto"
    )

    # pyLDAvis.enable_notebook()
    # vis = gensimvis.prepare(
    #     lda_model,
    #     corp,
    #     id2word,
    #     mds="mmds",
    #     R=20)

    # assign detected labels to every message
    for case in casesList:
        for message in case.messages:
            cor = message.subject + " " + message.content
            new_text_corpus = id2word.doc2bow(cor.split())
            highestPercentageLabel = max(lda_model[new_text_corpus], key=lambda x:x[1])[0]
            if (message.from_.split("@")[1] == message.to.split("@")[1]):
                message.detectedLabel = 3
            else:
                message.detectedLabel = highestPercentageLabel

    # get all train labels
    labels = casesList.getTrainLabels()
    # get amount of similar classifications for each label
    quotas = []
    print(f"Stats for n_topics={pool}:")
    print("---------------------------")
    for label in labels:
        # get all messages with label and check similar
        msg = casesList.getMessagesByLabel(label)
        detectedLabels = [m.detectedLabel for m in msg]
        try:
            max_ocurr = (max(sorted(set([i for i in detectedLabels if detectedLabels.count(i)>2]))))
        except:
            max_occur = 0
        num = detectedLabels.count(max_ocurr)
        rest = len(detectedLabels)-num
        quota = num/(rest+num)
        print(f"Correctly identified messages for label {label}: {num}/{num+rest}={quota}")
        quotas.append([quota, len(detectedLabels)])
    label_elcounts = [x[1] for x in quotas]
    labelscores = [x[0] for x in quotas]
    weighted = [a*b for a,b in zip(label_elcounts, labelscores)]
    overall = sum(weighted)/sum(label_elcounts)
    print(f"Overall result: {overall}")
    pool_scores.append((pool, overall, quotas))

#%%
# Save Parameter Study values for later plotting (change filename accordingly)
import pickle
filename = "lemma_simple_tfidf3.pkl"
p = path.join("out", "parameter_study")

with open(path.join(p, filename), "wb") as f:
    pickle.dump(pool_scores, f)