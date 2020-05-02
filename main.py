from nltk.corpus import twitter_samples
from nltk.tag import pos_tag
from nltk.corpus import twitter_samples
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix
import csv


import re, string
import random

##Déterminer les mots significatifs en anglais
stop_words = stopwords.words('english')


## Enlever les données non significatives
def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-z]|[A-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens




## Déterminer le type du mot dans une données
def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence


#positive_tweets = twitter_samples.strings('positive_tweets.json')
#negative_tweets = twitter_samples.strings('negative_tweets.json')

#tweet_tokens = twitter_samples.tokenized('positive_tweets.json')

## Les Références des mots positives/négatives dans des tweets
positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')
positive_cleaned_tokens_list = []
negative_cleaned_tokens_list = []

for tokens in positive_tweet_tokens:
    positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

for tokens in negative_tweet_tokens:
    negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))


##spécifier les mots les plus utilisées
#def get_all_words(cleaned_tokens_list):
#    for tokens in cleaned_tokens_list:
#        for token in tokens:
#            yield token

#all_pos_words = get_all_words(positive_cleaned_tokens_list)
#freq_dist_pos = FreqDist(all_pos_words)
#print(freq_dist_pos.most_common(10))


def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)


## si la donnée est positive, on retourne un 1 sinon un 0
positive_dataset = [(tweet_dict, "1")
                     for tweet_dict in positive_tokens_for_model]

negative_dataset = [(tweet_dict, "0")
                     for tweet_dict in negative_tokens_for_model]

dataset = positive_dataset + negative_dataset

random.shuffle(dataset)

train_data = dataset[:7000]
test_data = dataset[7000:]


classifier = NaiveBayesClassifier.train(train_data)

#print("Accuracy is:", classify.accuracy(classifier, test_data))

#print(classifier.show_most_informative_features(10))

arrR = [] 
arrP = []

with open('test.csv', encoding="utf8") as file:
    reader = csv.reader(file)
    
    for row in reader:
        custom_tweet = row[2]
        custom_tokens = remove_noise(word_tokenize(custom_tweet))
        print(row[1] +'           '+classifier.classify(dict([token, True] for token in custom_tokens)))
        arrP.append(classifier.classify(dict([token, True] for token in custom_tokens)))
        arrR.append(row[1])

tableDeConfusion = confusion_matrix(arrR,arrP)
print(tableDeConfusion)


#print(tweet_tokens[0])
#print(pos_tag(tweet_tokens[0]))
#print(lemmatize_sentence(tweet_tokens[0]))
#print(remove_noise(tweet_tokens[0], stop_words))


##Remove noise from positive tweets dataset
#print(positive_tweet_tokens[500])
#print(positive_cleaned_tokens_list[500])