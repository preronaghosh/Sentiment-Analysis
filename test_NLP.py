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

	x_test = []
	y_test = []

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

	# Load test data 
	test_pos_path = './data/aclImdb/test/pos/*.txt'
	test_neg_path = './data/aclImdb/test/neg/*.txt'

	# Populate the test dataset
	for file_path in glob.glob(test_pos_path):
		with open(file_path, 'r', encoding='utf-8') as file:
			x_test.append(file.read())
			y_test.append(1)
			
	for file_path in glob.glob(test_neg_path):
		with open(file_path, 'r', encoding='utf-8') as file:
			x_test.append(file.read())
			y_test.append(0)

	x_test_cleaned = cleanup_text(x_test)

	# Split the sentences into words and remove stopwords
	nltk.download('stopwords')
	stop_words = stopwords.words('english')

	tokenizer = RegexpTokenizer(r'\w+')

	X_test = []
	X_train = []
	
	for text in x_train_cleaned:
		tokens = tokenizer.tokenize(text)
		final_tokens = [tk for tk in tokens if tk not in stop_words]

		# Construct a sentence with the tokens and store in a new training dataset
		final_text = " ".join(final_tokens)
		X_train.append(final_text)

	for text in x_test_cleaned:
		tokens = tokenizer.tokenize(text)
		final_tokens = [tk for tk in tokens if tk not in stop_words]

		# Construct a sentence with the tokens and store in a new test dataset
		final_text = " ".join(final_tokens)
		X_test.append(final_text)

	# Applying Lemmatization
	nltk.download('punkt')
	nltk.download('omw-1.4')

	X_train_lem = []
	X_test_lem = []

	lemmatizer = WordNetLemmatizer()

	# Lemmatize training dataset
	for review in X_train:
		tokens = nltk.word_tokenize(review)
		lem_review = " ".join([lemmatizer.lemmatize(token) for token in tokens])
		X_train_lem.append(lem_review)
		
	X_train = X_train_lem

	# Lemmatize test dataset
	for review in X_test:
		tokens = nltk.word_tokenize(review)
		lem_review = " ".join([lemmatizer.lemmatize(token) for token in tokens])
		X_test_lem.append(lem_review)

	X_test = X_test_lem

	# Convert textual data to integer sequences
	tok = keras.preprocessing.text.Tokenizer()
	tok.fit_on_texts(X_train)
	X_test = tok.texts_to_sequences(X_test)

	# Pad the sequences as input is of varying lengths
	X_test = pad_sequences(X_test,padding='post',maxlen=1000)
	y_test = np.array(y_test)

	# Load the saved model
	model = tf.keras.models.load_model("./models/21048873_NLP_model.h5")

	# Run prediction on the test data and print the test accuracy
	y_pred = model.predict(X_test)
	y_pred = y_pred.reshape(-1,)
	y_pred_binary = np.round(y_pred).astype(int)

	print(f"Test Accuracy is: {accuracy_score(y_test, y_pred_binary) * 100}")
