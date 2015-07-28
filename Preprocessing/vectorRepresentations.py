import numpy
from nltk.util import ngrams
from sklearn.preprocessing import scale


# apply k-menas
def kmeansSentiment(clean_sent, kMeansModel, pcaModel, word2vecModel):

    params = kMeansModel.get_params()
    k = params['n_clusters']
    k_dict = dict.fromkeys(range(k), 0)

    for word in clean_sent.split():
        try:
            vector = pcaModel.transform(scale(word2vecModel[word]))
        except:
            continue
        prediction = kMeansModel.predict(vector)
        k_dict[prediction[0]] += 1
    k_dict_sorted = sorted(k_dict.items(), key=lambda x: x[0])
    return [c[1] for c in k_dict_sorted]


# derive ngram bagOfWords representation of a review
def buildBagOfNgrams(clean_sent, ngram_list):
    n = len(ngram_list[0])
    ngrams_dict = dict.fromkeys(ngram_list, 0)
    for ngram in ngrams(clean_sent, n):
        if ngram in ngrams_dict:
            ngrams_dict[ngram] = 1
    ngram_dict_sorted = sorted(ngrams_dict.items(), key=lambda x: x[0])
    return [c[1] for c in ngram_dict_sorted]


# derive sentence representation have sum of word vectors
def buildSentVecAsSum(clean_sent, model):

    temp = numpy.zeros((1,300))

    for w in clean_sent:
        try:
            temp = temp + model[w]
        except:
            pass

    return temp


# derive sentence representation as average of word vectors
def buildSentVecAsAverage(clean_sent, model):

    temp = numpy.zeros((1,300))
    N = 0

    for w in clean_sent:
        try:
            temp = temp + model[w]
            N = N+1
        except:
            pass
    if N>0:
        temp = temp/N

    return temp