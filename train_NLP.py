import glob
import numpy as np
import tensorflow as tf
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import keras 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.stem import WordNetLemmatizer

def cleanup_text(reviews):
    cleaned_reviews = []
    
    for review in reviews:
        review = re.sub(r'<.*?>', '', review)  # remove html tags first
        cleaned_text = re.sub(r'[^\w\s]', '', review)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text) # remove whitespace
        cleaned_text = cleaned_text.lower() # convert to lowercase
        cleaned_reviews.append(cleaned_text)
        
    return cleaned_reviews


if __name__ == "__main__": 

	x_train = []
	y_train = []

	pos_path = './data/aclImdb/train/pos/*.txt'
	neg_path = './data/aclImdb/train/neg/*.txt'

	# Populate the training dataset
	for file_path in glob.glob(pos_path):
		with open(file_path, 'r', encoding='utf-8') as file:
			x_train.append(file.read())
			y_train.append(1)
			
	for file_path in glob.glob(neg_path):
		with open(file_path, 'r', encoding='utf-8') as file:
			x_train.append(file.read())
			y_train.append(0)

	x_train_cleaned = cleanup_text(x_train)

	# Split the sentences into words and remove stopwords
	nltk.download('stopwords')
	stop_words = stopwords.words('english')

	tokenizer = RegexpTokenizer(r'\w+')

	X_train = []
	
	for text in x_train_cleaned:
		tokens = tokenizer.tokenize(text)
		final_tokens = [tk for tk in tokens if tk not in stop_words]

		# Construct a sentence with the tokens and store in a new training dataset
		final_text = " ".join(final_tokens)
		X_train.append(final_text)


	# Applying Lemmatization
	nltk.download('punkt')
	nltk.download('omw-1.4')

	X_train_lem = []

	lemmatizer = WordNetLemmatizer()

	# Lemmatize training dataset
	for review in X_train:
		tokens = nltk.word_tokenize(review)
		lem_review = " ".join([lemmatizer.lemmatize(token) for token in tokens])
		X_train_lem.append(lem_review)
		
	X_train = X_train_lem

	# Convert textual data to integer sequences
	tok = keras.preprocessing.text.Tokenizer()
	tok.fit_on_texts(X_train)
	X_train = tok.texts_to_sequences(X_train)

	# Pad the sequences as input is of varying lengths
	max_seq_len = 1000
	X_train = pad_sequences(X_train,padding='post',maxlen=max_seq_len)

	y_train = np.array(y_train)

	# Split dataset into 80% training and 20% validation sets
	X_train2, X_validation2, y_train2, y_validation2 = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

	vocab_size = len(tok.word_index) # stores the number of unique words

	# Build the model
	model = keras.Sequential()
	# Input layer of total vocabulary, each feature is a 16 dimensional vector 
	model.add(keras.layers.Embedding(vocab_size+1, 16, input_length=1000))  # add 1 to record for unknown words at index 0 
	model.add(keras.layers.Dropout(0.1))
	model.add(keras.layers.Conv1D(filters=16,kernel_size=2,padding='valid',activation='relu'))
	model.add(keras.layers.GlobalAveragePooling1D())
	model.add(keras.layers.Dropout(0.15))
	model.add(keras.layers.Dense(32, activation='tanh'))
	model.add(keras.layers.Dropout(0.15))
	model.add(keras.layers.Dense(1, activation='sigmoid'))

	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
	model.fit(X_train2, y_train2,
            epochs=10,
            validation_data=(X_validation2, y_validation2),
            verbose=1,
            batch_size=512)
	
	model.save('./models/21048873_NLP_model.h5')