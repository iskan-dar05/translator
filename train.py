
import tarfile
import os
import urllib.request
import re
from nltk.tokenize import word_tokenize
import nltk
import os, gc
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle


nltk.download('punkt_tab')



MODEL_PATH="nmt/model.keras"
TOKENIZER_PATH="nmt/tokenizer.pkl"
STATE_PATH="nmt/state.pkl"
DATASET_DIR = "dataset"
file1="fr-en.tgz"
file2="es-en.tgz"

def extract(file):
	try:
		with tarfile.open(file, "r:gz") as tar:
			tar.extractall(path=DATASET_DIR)
			print(f"Successfully extract file {file} to {DATASET_DIR}")
	except tarfile.TarError as e:
		print(f"An error occurred during extraction: {e}")
	except FileNotFoundError:
		print(f"File {file} not found.")



def load_extract_data():
	if not os.path.exists('fr-en.tgz'):
		urllib.request.urlretrieve('https://www.statmt.org/europarl/v7/fr-en.tgz', 'fr-en.tgz')
	
	if not os.path.exists('es-en.tgz'):
		urllib.request.urlretrieve('https://www.statmt.org/europarl/v7/es-en.tgz', 'es-en.tgz')

	if not os.path.exists(DATASET_DIR):
		os.makedirs(DATASET_DIR)

	


	extract(file1)
	extract(file2)




def clean(line):
	line = line.strip()

	line = re.sub(r'<.*?>', '', line)

	line = re.sub(r'[^a-zA-Z0-9\s]', '', line)

	line = re.sub(r'\d+', '', line)

	line = line.lower()

	return line







if __name__ == '__main__':
	load_extract_data()
	with open(DATASET_DIR + "/europarl-v7.fr-en.en", "r", encoding="utf-8") as en:
		en_fr_content = en.readlines()
	with open(DATASET_DIR + "/europarl-v7.fr-en.fr", "r", encoding="utf-8") as fr:
		fr_en_content = fr.readlines()
	with open(DATASET_DIR + "/europarl-v7.es-en.es", "r", encoding="utf-8") as es:
		es_en_content = es.readlines()
	with open(DATASET_DIR + "/europarl-v7.es-en.en", "r", encoding="utf-8") as en:
		en_es_content = en.readlines()
	en_fr_list = [clean(line) for line in en_fr_content]
	fr_en_list = [clean(line) for line in fr_en_content]
	es_en_list = [clean(line) for line in es_en_content]
	en_es_list = [clean(line) for line in en_es_content]
	pairs = []

	# EN → FR
	for en, fr in zip(en_fr_list, fr_en_list):
		src = "<en> <to_fr> " + en
		tgt = "<sos> " + fr + " <eos>"
		pairs.append((src, tgt))

	# FR → EN
	for en, fr in zip(en_fr_list, fr_en_list):
		src = "<fr> <to_en> " + fr
		tgt = "<sos> " + en + " <eos>"
		pairs.append((src, tgt))

	# EN -> ES
	for en, es in zip(en_es_list, es_en_list):
		src = "<en> <to_es> " + en
		tgt = "<sos> " + es + " <eos>"
		pairs.append((src, tgt))

	# ES -> EN
	for en, es in zip(en_es_list, es_en_list):
	    src = "<es> <to_en> " + es
	    tgt = "<sos> " + en + " <eos>"
	    pairs.append((src, tgt))

	from sklearn.model_selection import train_test_split

	train_pairs, test_pairs = train_test_split(pairs, test_size=0.2, random_state=42)


	from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
	from tensorflow.keras.models import Model
	import pickle
	if os.path.exists(TOKENIZER_PATH):
		with open(TOKENIZER_PATH, "rb") as f:
		    tokenizer = pickle.load(f)
	else:
		all_texts = [p[0] for p in pairs] + [p[1] for p in pairs]

		tokenizer = Tokenizer(num_words=30000, filters='')
		tokenizer.fit_on_texts(all_texts)

		# Ensure the directory exists before saving the tokenizer
		os.makedirs(os.path.dirname(TOKENIZER_PATH), exist_ok=True)

		with open(TOKENIZER_PATH, "wb") as f:
		    pickle.dump(tokenizer, f)

	vocab_size = len(tokenizer.word_index) + 1
	latent_dim = 128
	encoder_inputs = Input(shape=(None,))
	encoder_emb = Embedding(vocab_size, latent_dim, mask_zero=True)(encoder_inputs)
	_, state_h, state_c = LSTM(latent_dim, return_state=True)(encoder_emb)

	encoder_states = [state_h, state_c]

	decoder_inputs = Input(shape=(None,))
	decoder_emb = Embedding(vocab_size, latent_dim, mask_zero=True)(decoder_inputs)

	decoder_outputs, _, _ = LSTM(
	    latent_dim,
	    return_sequences=True,
	    return_state=True
	)(decoder_emb, initial_state=encoder_states)

	decoder_outputs = Dense(vocab_size, activation='softmax')(decoder_outputs)







	if os.path.exists(MODEL_PATH):
		model = load_model(MODEL_PATH)
		


	else:
		model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

		model.compile(
		    optimizer='adam',
		    loss='sparse_categorical_crossentropy',
		    metrics=['accuracy']
		)

		model.summary()



	def chunked(data, size):
	    for i in range(0, len(data), size):
	        yield data[i:i + size]



	def save_state(chunk_id):
	    with open(STATE_PATH, "wb") as f:
	        pickle.dump({"chunk_id": chunk_id}, f)

	def load_state():
	    if os.path.exists(STATE_PATH):
	        with open(STATE_PATH, "rb") as f:
	            return pickle.load(f)["chunk_id"]
	    return 0



	MAX_LEN = 40
	CHUNK_SIZE = 10000
	BATCH_SIZE = 16
	EPOCHS_PER_CHUNK = 1


	start_chunk = load_state()
	print("Resuming from chunk:", start_chunk)

	for chunk_id, chunk_pairs in enumerate(chunked(train_pairs, CHUNK_SIZE)):

	    if chunk_id < start_chunk:
	        continue   # ⚠️ skip already-trained chunks

	    print(f"\nTraining chunk {chunk_id}")

	    # Prepare data
	    encoder_texts = [p[0] for p in chunk_pairs]
	    decoder_texts = [p[1] for p in chunk_pairs]

	    encoder_seq = tokenizer.texts_to_sequences(encoder_texts)
	    decoder_seq = tokenizer.texts_to_sequences(decoder_texts)

	    encoder_seq = pad_sequences(encoder_seq, maxlen=MAX_LEN, padding="post")
	    decoder_seq = pad_sequences(decoder_seq, maxlen=MAX_LEN, padding="post")

	    decoder_in  = decoder_seq[:, :-1]
	    decoder_out = decoder_seq[:, 1:]
	    decoder_out = np.expand_dims(decoder_out, -1)

	    # Train
	    model.fit(
	        [encoder_seq, decoder_in],
	        decoder_out,
	        batch_size=BATCH_SIZE,
	        epochs=EPOCHS_PER_CHUNK
	    )

	    # Save model + state
	    model.save(MODEL_PATH)
	    save_state(chunk_id + 1)

	    # Free memory
	    del encoder_seq, decoder_seq, decoder_in, decoder_out
	    gc.collect()


