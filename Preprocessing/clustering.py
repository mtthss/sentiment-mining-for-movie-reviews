from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from gensim.models.word2vec import Word2Vec
from csv import reader
from time import time
from numpy import array
from pickle import dump


model_name = '../Data/GoogleNews-vectors-negative300.bin'
train_path = '../Data/train.tsv'
test_path = '../Data/test.tsv'


print '\nLoading Word2Vec model ...'
word2vecModel = Word2Vec.load_word2vec_format(model_name, binary=True)  # C binary format


print '\nReading the train and test datasets ...'
dataset = []
with open(test_path) as test, open(train_path) as train:
    tr = reader(train, dialect='excel-tab')
    te = reader(test, dialect='excel-tab')
    tr.next()
    te.next()
    for row in tr:
        dataset.extend(map(lambda x: x.lower(), row[2].split()))
    for row in te:
        dataset.extend(map(lambda x: x.lower(), row[2].split()))


print '\nTransforming sentences into word2vec representation ...'
vectors = []
for word in set(dataset):
    try:
        vectors.append(word2vecModel[word])
    except:
        pass


print '\nApplying PCA ...'
vectors = array(vectors)
pca = PCA(n_components='mle')
vectors = pca.fit_transform(vectors)
print('Number of PCA dimensions chosen: %d' % pca.n_components_)


print '\nApplying kMeans ...'
start_time = time()
kMeansModel = KMeans(init='k-means++', n_clusters=8, n_init=10)
kMeansModel.fit(vectors)
train_time = time() - start_time
print('kmeans took %f seconds' % train_time)


print '\nPickling ...'
dump(kMeansModel, open("../Data/kMeans8.pkl", "wb"))
