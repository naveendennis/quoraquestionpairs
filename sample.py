import pandas as pd
import os.path
import numpy as np
import math
import gensim
from nltk.corpus import stopwords
import string
import pickle
import nltk
from textblob import Sentence
from numpy.core.defchararray import *
from fuzzywuzzy import fuzz


class PhraseVector:
    def __init__(self, phrase):
        self.phrase = phrase
        self.vector = self.phrase_to_vec(phrase)
        self.pos_tag = self.get_words_in_phrase(phrase)

    @staticmethod
    def convert_vector_set_to_average(vector_set, ignore=[]):
        if len(ignore) == 0:
            return np.mean(vector_set, axis=0)
        else:
            return np.dot(np.transpose(vector_set), ignore) / sum(ignore)

    @staticmethod
    def get_unique_token_tags(vector1, vector2):
        tag_list = []
        for each_tag in vector1.pos_tag + vector2.pos_tag:
            if each_tag not in tag_list:
                tag_list.append(each_tag)
        return tag_list

    def phrase_to_vec(self, phrase):
        _stop_words = stopwords.words("english")
        phrase = phrase.lower()
        verified_words = [word for word in phrase.split() if word not in _stop_words]
        vector_set = []
        for each_word in verified_words:
            try:
                word_vector = word_model[each_word]
                vector_set.append(word_vector)
            except:
                pass
        return self.convert_vector_set_to_average(vector_set)

    def get_cosine_similarity(self, other_vector):
        cosine_similarity = np.dot(self.vector, other_vector.vector) / (
        np.linalg.norm(self.vector) * np.linalg.norm(other_vector.vector))
        try:
            if math.isnan(cosine_similarity):
                cosine_similarity = 0
        except:
            cosine_similarity = 0
        return cosine_similarity

    def get_words_in_phrase(self, phrase):
        if phrase.strip() == '':
            return []
        else:
            tagged_input = nltk.pos_tag(phrase.split(), tagset='universal')
            prev_item, prev_tag = tagged_input[0]
            g_item_list = [prev_item]
            cur_group_index = 0
            space = ' '
            revised_tag = []
            for cur_item, cur_tag in tagged_input[1:]:
                cur_item = cur_item.lower()
                if prev_tag is cur_tag:
                    g_item_list[cur_group_index] += space + cur_item
                else:
                    revised_tag.append((g_item_list[cur_group_index], prev_tag))
                    prev_tag = cur_tag
                    g_item_list.append(cur_item)
                    cur_group_index += 1
            revised_tag.append((g_item_list[cur_group_index], prev_tag))
            return revised_tag

    def get_token_similarity(self, other_vector):
        try:
            unique_tokens = PhraseVector.get_unique_token_tags(self, other_vector)
            len_unique_tokens = len(unique_tokens)
            matches = 0
            for each_tag in self.pos_tag:
                if each_tag in other_vector.pos_tag:
                    matches += 1
            return matches / len_unique_tokens
        except:
            return 0


def get_phrase_vector_obj(value):
    return PhraseVector(value)


def get_cosine_similarity(vector1, vector2):
    return vector1.get_cosine_similarity(vector2)


def get_token_similarity(vector1, vector2):
    return vector1.get_token_similarity(vector2)


def get_predict_score(w2v_score):
    return 1 if (w2v_score) > THRESHOLD_VALUE else 0


def get_cosine_similarity_vector(vector1, vector2):
    similarity_vector = []
    for each_v1, each_v2 in zip(vector1, vector2):
        similarity_vector.append(get_cosine_similarity(each_v1, each_v2))
    return np.array(similarity_vector)


def get_fuzzy_partial_vector(phrase_obj):
    fuzzy_partial_array = []
    for each in phrase_obj:
        if each is not None and type(each) is str:
            fuzzy_partial_array.append(fuzz.partial_ratio(each))
        else:
            fuzzy_partial_array.append(0)
    return np.array(fuzzy_partial_array)


def get_phrase_vector(phrase_obj):
    phrase_vector = []
    for each in phrase_obj:
        if type(each.vector) is np.ndarray:
            phrase_vector = each.vector.tolist()+phrase_vector
        else:
            phrase_vector = [13]*300 + phrase_vector
    return np.array([phrase_vector])


def get_features(features, operation='train'):
    row = features.shape[0]
    phrase_vectors1 = translate(features[:, 0].astype(str), table=translator)
    phrase_vectors2 = translate(features[:, 1].astype(str), table=translator)

    filename = os.path.join(dir_path, 'data','sentiment_vectors_'+operation)
    if not os.path.exists(filename):
        sentiment_vector1 = np.array([Sentence(each).polarity for each in phrase_vectors1]).reshape(row, 1)
        sentiment_vector2 = np.array([Sentence(each).polarity for each in phrase_vectors2]).reshape(row, 1)
        with open(filename, 'wb') as f:
            pickle.dump(sentiment_vector1, f)
            pickle.dump(sentiment_vector2, f)
    else:
        with open(filename, 'rb') as f:
            sentiment_vector1 = pickle.load(f)
            sentiment_vector2 = pickle.load(f)

    filename = os.path.join(dir_path, 'data','subjective_vectors_'+operation)
    if not os.path.exists(filename):
        subjective_vectors1 = np.array([Sentence(each).subjectivity for each in phrase_vectors1]).reshape(row, 1)
        subjective_vectors2 = np.array([Sentence(each).subjectivity for each in phrase_vectors2]).reshape(row, 1)
        with open(filename, 'wb') as f:
            pickle.dump(subjective_vectors1, f)
            pickle.dump(subjective_vectors2, f)
    else:
        with open(filename, 'rb') as f:
            subjective_vectors1 = pickle.load(f)
            subjective_vectors2 = pickle.load(f)

    filename = os.path.join(dir_path, 'data','fuzzy_wuzzy_partial_ratio_'+operation)
    if not os.path.exists(filename):
        partial_ratio_vector1 = get_fuzzy_partial_vector(phrase_vectors1).reshape(row, 1)
        partial_ratio_vector2= get_fuzzy_partial_vector(phrase_vectors2).reshape(row, 1)
        with open(filename, 'wb') as f:
            pickle.dump(partial_ratio_vector1, f)
            pickle.dump(partial_ratio_vector2, f)
    else:
        with open(filename, 'rb') as f:
            partial_ratio_vector1 = pickle.load(f)
            partial_ratio_vector2 = pickle.load(f)

    filename = os.path.join(dir_path, 'data', 'raw_phrase_vectors_'+operation)
    if not os.path.exists(filename):
        phrase_vectors1 = np.vectorize(get_phrase_vector_obj)(phrase_vectors1)
        phrase_vectors2 = np.vectorize(get_phrase_vector_obj)(phrase_vectors2)
        with open(filename, 'wb') as f:
            pickle.dump(phrase_vectors1, f)
            pickle.dump(phrase_vectors2, f)
    else:
        with open(filename, 'rb') as f:
            phrase_vectors1 = pickle.load(f)
            phrase_vectors2 = pickle.load(f)

    filename = os.path.join(dir_path, 'data','cosine_similarity_vector_'+operation)
    if not os.path.exists(filename):
        cosine_similarity_vector = get_cosine_similarity_vector(phrase_vectors1, phrase_vectors2).reshape(row, 1)
        with open(filename, 'wb') as f:
            pickle.dump(cosine_similarity_vector, f)
    else:
        with open(filename, 'rb') as f:
            cosine_similarity_vector = pickle.load(f)

    filename = os.path.join(dir_path, 'data', 'processed_phrase_vectors_'+operation)
    if not os.path.exists(filename):
        phrase_vectors1 = get_phrase_vector(phrase_vectors1).reshape(row, 300)
        phrase_vectors2 = get_phrase_vector(phrase_vectors2).reshape(row, 300)
        with open(filename, 'wb') as f:
            pickle.dump(phrase_vectors1, f)
            pickle.dump(phrase_vectors2, f)
    else:
        with open(filename, 'rb') as f:
            phrase_vectors1 = pickle.load(f)
            phrase_vectors2 = pickle.load(f)

    features = np.concatenate((cosine_similarity_vector, partial_ratio_vector1, partial_ratio_vector2, subjective_vectors1, subjective_vectors2, sentiment_vector1, sentiment_vector2, phrase_vectors1, phrase_vectors2), axis=1)
    return features


if __name__ == '__main__':
    dir_name = os.path.dirname(os.path.realpath(__file__))
    TRAIN_FILE = os.path.join(dir_name , 'dataset','train.csv')
    TEST_FILE = os.path.join(dir_name,  'dataset','test.csv')
    THRESHOLD_VALUE = 0.8
    translator = str.maketrans(' ', ' ', string.punctuation)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_filename = os.path.join(dir_name ,'data','google_p2v_model')

    if not os.path.exists(model_filename):
        pathToBinVectors = os.path.join(dir_name,'data','GoogleNews-vectors-negative300.bin')
        print("Loading the data file... Please wait...")
        word_model = gensim.models.KeyedVectors.load_word2vec_format(pathToBinVectors, binary=True)
        print("Successfully loaded 3.6 G bin file!")
        pickle.dump(word_model, open(model_filename, 'wb'))
    else:
        word_model = pickle.load(open(model_filename, 'rb'))
        print('Successfully Loaded the model')

    train_contents = pd.read_csv(TRAIN_FILE)
    train_contents = pd.np.array(train_contents)
    header = train_contents[0]
    train_contents = train_contents[1:80001]
    labels = train_contents[:, 5].astype(int)
    features = train_contents[:, 3:5]
    from sklearn.model_selection import train_test_split
    feature_train, feature_test, label_train, label_test = train_test_split(features, labels, train_size=0.50)
    feature_train = get_features(features=feature_train)
    filename = os.path.join(dir_path , 'data', 'svm_plain')
    if not os.path.exists(filename):
        from sklearn.svm import SVC
        clf = SVC()
        with open(filename, 'wb') as f:
            pickle.dump(clf, f)
    else:
        with open(filename, 'rb') as f:
            clf = pickle.load(f)

    feature_test = get_features(features=feature_test, operation='test')
    from sklearn.model_selection import cross_val_score
    accuracy = cross_val_score(clf, feature_test, label_test)
    print('Accuracy: ', accuracy)
