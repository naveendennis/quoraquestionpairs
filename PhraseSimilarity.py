######################
# Loading word2vec
######################

import gensim
import os
import pickle
import numpy as np
import math
from nltk.corpus import stopwords
import nltk

dir_name = os.path.dirname(os.path.realpath(__file__))
model_filename = dir_name+'/data/google_p2v_model'
if not os.path.exists(model_filename):
    pathToBinVectors = 'data/GoogleNews-vectors-negative300.bin'
    print("Loading the data file... Please wait...")
    model1 = gensim.models.KeyedVectors.load_word2vec_format(pathToBinVectors, binary=True)
    print("Successfully loaded 3.6 G bin file!")
    pickle.dump(model1, open(model_filename, 'wb'))
else:
    model1 = pickle.load(open(model_filename, 'rb'))
    print('Successfully Loaded the model')

class PhraseVector:
	def __init__(self, phrase):
		self.vector = self.PhraseToVec(phrase)
		self.pos_tag = self.get_words_in_phrase(phrase)

	def ConvertVectorSetToVecAverageBased(self, vectorSet, ignore = []):
		if len(ignore) == 0:
			return np.mean(vectorSet, axis = 0)
		else:
			return np.dot(np.transpose(vectorSet),ignore)/sum(ignore)


	def PhraseToVec(self, phrase):
		cachedStopWords = stopwords.words("english")
		phrase = phrase.lower()
		wordsInPhrase = [word for word in phrase.split() if word not in cachedStopWords]
		# wordsInPhrase = self.get_words_in_phrase(phrase)
		# print(wordsInPhrase)
		vectorSet = []
		for aWord in wordsInPhrase:
			try:
				wordVector=model1[aWord]
				vectorSet.append(wordVector)
			except:
				pass
		return self.ConvertVectorSetToVecAverageBased(vectorSet)

	def CosineSimilarity(self, otherPhraseVec):
		cosine_similarity = np.dot(self.vector, otherPhraseVec) / (np.linalg.norm(self.vector) * np.linalg.norm(otherPhraseVec))
		try:
			if math.isnan(cosine_similarity):
				cosine_similarity=0
		except:
			cosine_similarity=0
		return cosine_similarity


	def get_words_in_phrase(self, phrase):
		tagged_input = nltk.pos_tag(phrase.split(), tagset='universal')
		prev_item, prev_tag = tagged_input[0]
		g_item_list = [prev_item]
		cur_group_index = 0
		space = ' '
		revised_tag= []
		for cur_item, cur_tag in tagged_input[1:]:
			if prev_tag is cur_tag:
				g_item_list[cur_group_index] += space + cur_item
				breaker = True
			else:
				revised_tag.append((g_item_list[cur_group_index], prev_tag))
				prev_tag = cur_tag
				g_item_list.append(cur_item)
				cur_group_index += 1
				breaker = False
		if breaker:
			revised_tag.append((g_item_list[cur_group_index], prev_tag))
		return [(each_token, each_type) for each_token, each_type in revised_tag if each_token.lower() not in stopwords.words('english')]
	# return [each for each in g_item_list if each.lower() not in stopwords.words('english')]

if __name__ == "__main__":
	print ("###################################################################")
	print ("###################################################################")
	print ("########### WELCOME TO THE PHRASE SIMILARITY CALCULATOR ###########" )
	print ("###################################################################")
	print ("###################################################################")

	if True:
		userInput1 = input("Type the phrase1: ")
		userInput2 = input("Type the phrase2: ")

		phraseVector1 = PhraseVector(userInput1)
		phraseVector2 = PhraseVector(userInput2)
		similarityScore  = phraseVector1.CosineSimilarity(phraseVector2.vector)
		print ("###################################################################")
		print ("Similarity Score: ", similarityScore)
		print ("###################################################################")
