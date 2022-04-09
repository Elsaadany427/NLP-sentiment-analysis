import pandas as pd

# importing dataset
from tqdm import tqdm

dataset = pd.read_csv('dataset.csv')
# for removing HTML Tags from text
from bs4 import BeautifulSoup
# for Removing Alphanumeric Text and Special Characters
import re
import nltk

# downloading stopwords
nltk.download('stopwords')

# import stop words
from nltk.corpus import stopwords
# import lemmatizer to returns an actual word of the language
from nltk.stem.wordnet import WordNetLemmatizer


# method to handle Contractions
def handleContractions(review):
    phrase = re.sub(r"won't", "will not", review)
    phrase = re.sub(r"mustn't", "must not", review)
    phrase = re.sub(r"must've", "must have", review)
    phrase = re.sub(r"needn't", "need not", review)
    phrase = re.sub(r"shouldn't", "should not", review)
    phrase = re.sub(r"should've", "should have", review)
    phrase = re.sub(r"weren't", "were not", review)
    phrase = re.sub(r"can\'t", "can not", review)
    phrase = re.sub(r"n\'t", " not", review)
    phrase = re.sub(r"\'re", " are", review)
    phrase = re.sub(r"\'s", " is", review)
    phrase = re.sub(r"\'d", " would", review)
    phrase = re.sub(r"\'ll", " will", review)
    phrase = re.sub(r"\'t", " not", review)
    phrase = re.sub(r"\'ve", " have", review)
    phrase = re.sub(r"\'m", " am", review)
    return phrase


# removing HTML tags
def removeHTMLTags(review):
    soup = BeautifulSoup(review, "lxml")
    return soup.get_text()


# remove Special Characters
def removeSpecialChars(review):
    return re.sub('[^a-zA-Z]', ' ', review)


# remove alphaNumeric words
def removeAlphaNumeric(review):
    return re.sub("\S*\d\S*", "", review).strip()


# preprocessing
def doCleaning(review):
    review = removeHTMLTags(review)
    review = removeAlphaNumeric(review)
    review = removeSpecialChars(review)
    review = handleContractions(review)

    # convert all words to lower case
    review = review.lower()

    # make Tokenization by splitting all with white space
    review = review.split()

    # removing stop words and make it Lemmatization
    lmtz = WordNetLemmatizer()
    # v is verb, n is noun
    review = [lmtz.lemmatize(word, 'v') for word in review if not (word in set(stopwords.words('english')))]
    review = " ".join(review)
    return review


# creating corpus
corpus = []
for idx, row in tqdm(dataset.iterrows()):
    review = doCleaning(row['Text'])
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer

# Creating the transform with Tri-gram
triGram = CountVectorizer(ngram_range=(1, 3), max_features=2)
# convert it to array of sequence of three words
X = triGram.fit_transform(corpus).toarray()
# second column of data frame (Number of users who indicated whether they found the review helpful or not)
y = dataset.iloc[:, 6].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB

# Creating Naive Bayes classifier
classifier = GaussianNB()

# Fitting the training set into the Naive Bayes classifier
classifier.fit(X_train, y_train)


# Predict sentiment for new Review
def predictNewReview():
    newReview = input("Type the Review: ")

    if newReview == '':
        print('Invalid Review')
    else:
        newReview = doCleaning(newReview)
        reviewVector = triGram.transform([newReview]).toarray()
        prediction = classifier.predict(reviewVector)
        print(prediction[0])
        if prediction[0] == 2:
            print("Positive Review")
        else:
            print("Negative Review")


predictNewReview()
