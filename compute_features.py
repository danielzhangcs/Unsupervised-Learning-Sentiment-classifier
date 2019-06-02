import gensim
import gensim.utils
import numpy as np
import pickle
from gensim.models.word2vec import Word2Vec
import utils
from sklearn.feature_extraction.text import TfidfVectorizer


#Change this if necessary
data_dir = 'data'
pretrained_word_embeddings_path = 'glove.6B.100d.word2vec_format.txt'

def get_reviews_embedding_from_word_embedding(reviews_tokens, model):
    review_embeddings = []
    for review_tokens in reviews_tokens:
        review_vectors = [model[token] for token in review_tokens if token in model]
        avg = np.average(review_vectors, axis=0) if len(review_vectors) is not 0 else np.zeros(model['good'].shape[0])
        review_embeddings.append(avg)
    return np.array(review_embeddings)


def get_custom_word_embeddings(reviews_tokens):
    model = Word2Vec(reviews_tokens, size=200, window=5, min_count=3, workers=4)
    word_vectors = model.wv
    del model
    return word_vectors


def get_custom_reviews_embeddings(reviews_tokens):
    model = get_custom_word_embeddings(reviews)
    return get_reviews_embedding_from_word_embedding(reviews_tokens, model)

def get_sparse_features(reviews):
    vectorizer = TfidfVectorizer(stopwords= 'english', ngram_range=(1,2))
    return vectorizer.fit_transform(reviews)


def save(vectors, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(vectors, f)


if __name__ == '__main__':
    print("Reading data")
    indices, reviews, labels = utils.get_training_data(data_dir)

    print("Pretrained Review Embeddings")
    reviews_tokens = [gensim.utils.simple_preprocess(review) for review in reviews]
    model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_word_embeddings_path, binary=False)
    save(get_reviews_embedding_from_word_embedding(reviews_tokens, model), 'reviews_features_dense_pretrained.pkl')

    print("Custom Word Embeddings")
    custom_word_embeddings = get_custom_word_embeddings(reviews_tokens)
    save(custom_word_embeddings, 'custom_word_embeddings.pkl')

    print("Custom Review Embeddings")
    save(get_reviews_embedding_from_word_embedding(reviews_tokens, custom_word_embeddings), 'reviews_features_dense_custom.pkl')

    print("Sparse Review Vectors")
    save(get_sparse_features(reviews), 'reviews_features_sparse.pkl')
