import time
import numpy
import csv
import cPickle

from csv import reader, writer
from gensim.models.word2vec import Word2Vec
from nltk import word_tokenize
from nltk.corpus import stopwords
from vectorRepresentations import buildSentVecAsSum, buildBagOfNgrams, buildSentVecAsAverage
from sentenceCleaning import cleanItUp
from pickle import load


# sentence representation
sentenceVect = buildSentVecAsSum # buildSentVecAsAverage

# initialize
trainset={}
testset={}
freq_nGrams_path = '../Data/frequentNgrams.pkl'
model_name = '../Data/GoogleNews-vectors-negative300.bin'
train_path = '../Data/train.tsv'
test_path = '../Data/test.tsv'
first_id = 156061
feat_word2vec = 300
feat_kMeans = 5
feat_nGrams = 270
feat_total = feat_word2vec + feat_nGrams + 1

# Load stopping/removal lists
cachedStopWords = stopwords.words("english")
removalList = stopwords.words("english")
removalList = removalList + ["'",",",".","ca'","n't","'s","-lrb-","-rrb-","''", '``']

# Load google pre-trained word vectors
print '\nLoading word2vec model...'
start_time = time.time()
model = Word2Vec.load_word2vec_format(model_name, binary=True)  # C binary format
model_time = time.time() - start_time

# Load Frequent nGrams
print '\nLoading Frequent nGrams...'
f = open(freq_nGrams_path, 'r')
freq_mGrams_list = cPickle.load(f)

# Load kMeans and PCA models
print '\nLoading PCA and kMeans models...'
kMeansModel = load(open("../Data/kMeans5.pkl", "rb" ))
pcaModel = load(open("../Data/pcaMLE.pkl", "rb" ))

# Load competition data
print '\nLoading competition data...'
start_time = time.time()

with open(train_path) as tsv:
    r = reader(tsv, dialect="excel-tab")
    r.next()
    for line in r:
        trainset[line[0]]=line[1:4]

reviewLoading_time_train = time.time() - start_time
start_time = time.time()

with open(test_path) as tsv:
    r = reader(tsv, dialect="excel-tab")
    r.next()

    for line in r:
        testset[line[0]]=line[1:4]

reviewLoading_time_test = time.time() - start_time

# Vectorize train sentences
print '\nVectorizing train sentences'
start_time = time.time()

data_matrix = numpy.zeros((len(trainset), feat_total+1))
for i in xrange(0, len(trainset)):

    sent = trainset[str(i+1)][1]
    curr_sent = word_tokenize(sent)
    clean_sent = [word for word in curr_sent if word not in cachedStopWords]
    data_matrix[i, 0] = int(trainset[str(i+1)][2])
    data_matrix[i, 1] = len(curr_sent)
    data_matrix[i, 2:feat_word2vec+2] = sentenceVect(clean_sent, model)
    data_matrix[i, feat_word2vec+2:feat_word2vec+feat_nGrams+2] = buildBagOfNgrams(cleanItUp(sent, removalList, False, False),freq_mGrams_list)

with open('../Data/transformedData.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for row in data_matrix.tolist():
        writer.writerow(row)

sentVect_time_train = time.time() - start_time

# Vectorize test sentences
print '\nVectorizing test sentences'
start_time = time.time()

data_matrix = numpy.zeros((len(testset), feat_total+1))
for i in xrange(first_id,first_id+len(testset)):

    sent = testset[str(i)][1]
    curr_sent = word_tokenize(sent)
    clean_sent = [word for word in curr_sent if word not in cachedStopWords]
    data_matrix[i-first_id, 0] = 2
    data_matrix[i-first_id, 1] = len(curr_sent)
    data_matrix[i-first_id, 1:feat_word2vec+1] = sentenceVect(clean_sent, model)
    data_matrix[i-first_id, feat_word2vec+1:feat_word2vec+feat_nGrams+1] = buildBagOfNgrams(cleanItUp(sent, removalList, False, False),freq_mGrams_list)

with open('../Data/transformedTestData.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for row in data_matrix.tolist():
        writer.writerow(row)

sentVect_time_test = time.time() - start_time

# Profiling
print '\n------------------'
print "Printing...\n"
print "Loading word2vec model: %f seconds" % model_time
print "Loading train movie reviews: %f seconds" % reviewLoading_time_train
print "Loading test movie reviews: %f seconds" % reviewLoading_time_test
print "Vectorizing train sentences: %f seconds" % sentVect_time_train
print "Vectorizing test sentences: %f seconds" % sentVect_time_test

# Ending
print '\n------------------'
print "Done!"