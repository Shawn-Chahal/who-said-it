import os
import pickle
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import VotingClassifier


def clean_text(text):
    text = re.sub(r"'", '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'pic.twitter\S+', '', text)
    text = re.sub(r'\W+', ' ', text.lower())

    return text


df = pd.read_csv(os.path.join('tweets', 'tweets.csv'), low_memory=False)
df.drop_duplicates(inplace=True)
df['tweet-clean'] = df['tweet'].apply(clean_text)
drop_index = []

for i in range(len(df)):
    if df['tweet-clean'].iloc[i] in ('', ' '):
        drop_index.append(i)

df.drop(drop_index, inplace=True)

random_state = 0
n_jobs = -1

tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
le = LabelEncoder()

X = tfidf.fit_transform(df['tweet-clean'])
y = le.fit_transform(df['name'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=random_state, stratify=y)

clf_log = LogisticRegression(C=20, solver='saga', random_state=random_state, n_jobs=n_jobs)
clf_bnb = BernoulliNB(alpha=0.01, binarize=0.09)
clf_vot = VotingClassifier(estimators=[('log', clf_log), ('bnb', clf_bnb)],
                           voting='soft', weights=(0.6, 0.4), n_jobs=n_jobs)

clf_log.fit(X_train, y_train)
clf_bnb.fit(X_train, y_train)
clf_vot.fit(X_train, y_train)

with open(os.path.join('logs', 'test_log.txt'), 'w') as f:
    f.write('Number of features:   {}\n'.format(X_train.shape[1]))
    f.write('Logistic accuracy:    {:.3f}\n'.format(clf_log.score(X_test, y_test)))
    f.write('Naive Bayes accuracy: {:.3f}\n'.format(clf_bnb.score(X_test, y_test)))
    f.write('Voting accuracy:      {:.3f}\n'.format(clf_vot.score(X_test, y_test)))

print('Please wait. Classifier is being trained...')

clf_vot.fit(X, y)

pickle.dump(clf_vot, open(os.path.join('pkl', 'clf.pkl'), 'wb'))
pickle.dump(tfidf, open(os.path.join('pkl', 'tfidf.pkl'), 'wb'))
pickle.dump(le, open(os.path.join('pkl', 'le.pkl'), 'wb'))

print('Classifier training complete!')

