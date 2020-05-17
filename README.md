# Who said it?

This repository trains the model which is used in the [Who said it?](https://www.shawnchahal.com/who-said-it) web app.

## What does the app do?

[Who said it?](https://www.shawnchahal.com/who-said-it) is a web app where the user writes a message and finds out whether Bernie Sanders or Donald Trump is more likely to say it. The model is trained by calculating the term frequency–inverse document frequency (tf-idf) vectorization of both politicians tweets. The tweets were obtained using [twint](https://github.com/twintproject/twint). A logistic regression and naive Bayes classifier are then trained and combined in a soft voting ensemble method using [scikit-learn](https://github.com/scikit-learn/scikit-learn).
