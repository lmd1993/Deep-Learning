import codecs
import pickle
#from word2vec.keras import global_settings as G
import global_settings as G

def write_array_to_file(wf, array):
	for i in range(len(array)):
		wf.write(str(array.item(i)) + " ")
	wf.write("\n")
def write_array_to_file_binary(wf, array):
	for i in range(len(array)):
		pickle.dump((str(array.item(i)) + " "), wf)
		# wf.write(bytes(chr(str(array.item(i)) + " ")))
	pickle.dump(("\n"), wf)
	# wf.write(bytes(chr("\n")))

def save_embeddings(save_filepath, weights, vocabulary):
	with codecs.open(save_filepath, "w", "utf-8") as wf:
		# First line is vocabulary size and embedding dimension
		wf.write(str(len(vocabulary)) + " " + str(weights.shape[1]) + "\n")
		# Now each line is word "\t" and embedding
		# First word is UNKNOWN_WORD by our convention
		index = 1
		wf.write(G.UNKNOWN_WORD + "\t")
		write_array_to_file(wf, weights[index])
		index += 1
		# Now emit embedding for each word based on their sorted counts
		sorted_words = reversed(sorted(vocabulary, key=lambda word: vocabulary[word]))
		for word in sorted_words:
			if word == G.UNKNOWN_WORD:
				continue
			wf.write(word + "\t")
			write_array_to_file(wf, weights[index])
			index += 1
def save_embeddings_binary(save_filepath, weights, vocabulary):
	with codecs.open(save_filepath, "wb") as wf:
		# First line is vocabulary size and embedding dimension
		pickle.dump((str(len(vocabulary)) + " " + str(weights.shape[1]) + "\n"), wf)
		# wf.write(bytes(chr(str(len(vocabulary)) + " " + str(weights.shape[1]) + "\n")))
		# Now each line is word "\t" and embedding
		# First word is UNKNOWN_WORD by our convention
		index = 1
		# wf.write(bytes(chr(G.UNKNOWN_WORD + "\t")))
		pickle.dump((G.UNKNOWN_WORD + "\t"), wf)
		write_array_to_file_binary(wf, weights[index])
		index += 1
		# Now emit embedding for each word based on their sorted counts
		sorted_words = reversed(sorted(vocabulary, key=lambda word: vocabulary[word]))
		for word in sorted_words:
			if word == G.UNKNOWN_WORD:
				continue
			# wf.write(bytes(chr((word + "\t"))))
			pickle.dump((word + "\t"), wf)
			write_array_to_file_binary(wf, weights[index])
			index += 1
