Restrictions:
* Local system was running a Windows 10
* Solution was computed on a machine without cuda support for GPU training and a maximum of 16GB RAM
* Tested on Windows 10 on a WSL2 Ubuntu setup

# Approach
I started out loading the data by transforming them into a basic pandas DataFrame and performing some basic data
analysis. I restriced myself at first only to `category` and `headline` as the task was to only predict the news
category using the `headline` of the article.

**_NOTE:_**  I later also tried including the `short_description` into the training data to try and improve the features
extracted from the text by adding more relevant data. I discarded the idea after some initial tests showed no
significant improvement of the model(s) (and the additional data slowed training down a lot)

**_NOTE:_** I also tried my algorithms on a different data set with fewer classes that I found during my research.

# Data Exploration
The basic data analysis included checking the min/max values of the data to see if the data was actually in the
mentioned time frame. Checking for NA values and empty strings. The data itself seemed relatively clean for `headline`
and `category`.

The classes are not distributed evenly and with the instruction of randomly sampling there is expected to be some
overfitting on the better represented classes like `POLITICS` and `WELLNESS`. In total there are 41 classes in total.
There are also some overlapping categories, which makes classification much harder on the data set.

Example: `POLITICS` and `WORLD NEWS`

I did some headline word count analysis to see how I would need to pad or crop my headlines to fit it into a basic NN or
CNN. I did not consider the RNN because I have only worked with one so far and it took me a long time to set up. I also
scrapped this idea to go for an easier solution using spacy later on.

# Data Cleaning
During the data cleaning I dropped everything but the `headline` and `category`. Furthermore I dropped all empty string
headlines, which were 6 samples out 200k+ rows. I removed all special characters and multiple whitespaces. Furthermore,
I transformed all headlines to lowercase.

# Model Exploration
I tried some approaches that worked well on other news datasets. Initially I started out using tfidf for some simple
feature extraction to see if the results would already be up to the requested > 90% accuracy goal. The Naive Bayes
classifier only reached around 55% accuracy.

**_NOTE:_** Right now the validation and test set are both used to compute some accuracy score. During development I
used the validation set for hyperparameter optimization, which was basically a single loop over a few configurations.
I removed these in the end to have a simpler evaluation setup in the code. (I could not try a lot of hyperparameter
configurations due to the limited RAM and long computation times of some approaches)

I continued building on the tfidf approach by trying out some different classifiers (LinearSVC, RandomForestClassifier,
LogisticRegression) to see if I get some better results. I couldn't get the basic methods to improve the results a lot.

This is when I decided to try out more complicated approach using neural networks. Due to the before mentioned hardware
limitations I stuck with spacy as it is easy to set up without having to do a lot of preprocessing and feature
extraction.

**_NOTE:_** I would have also tried BERT and some other NLP approaches if I had the time and hardware as it works really
well for NLP.

I tested spacy with some simple BoW and tfidf and maxed out at around 59%-60% accuracy. Here I also had a look at the
classification report for the different classes and as expected, the algorithm only performed well on the well
represented classes. The recall and precision for the other classes is often times very low.

During my research I also had a look at the discussions on the kaggle dataset and seemingly an accuracy around 60-70%
was a good result being able to use all the data for the prediction.

**_NOTE:_** I also tried my methods on a smaller dataset with less categories to see how it performs on other data. The
dataset can be found here (https://www.kaggle.com/uciml/news-aggregator-dataset) but is also included in the repository.
On this set all my methods performed much better with accuracies up to 95%.

I tried some additional changes which deviated from the task to see if I could quickly improve the results. I tried
stratifying the train test split, which did not have a huge effect. If I had more time I would have also tried some
other approaches to counter problem of the overrepresented categories.

# Additional Work
Since I could not reach the expected > 90%, I added some additional code which I created to actually make the model that
I build deployable. For this I took the following steps:

* Adapted the throw-away notebook code to deployable code
* Dockerize the application
* Build a simple API using flask
* Create a docker compose file for training
* Create a docker compose file to run the API

In order to separate the training from the execution I had to set up a training pipeline and an execution pipeline. In
machine learning training and execution share the preprocessing steps, so I wrote the preprocessing steps to be usable
by both pipelines.

The trained model had to be stored somewhere as I did not want to have to retrain a model every time I run my API.
Luckily spacy simply allows you to pickle the trained model.

**_NOTE:_** A small problem occured due to not having a proper Linux setup at home. When you pickle a spacy model in
Windows it apparently pickles some path in some Windows format. Thus when I tried to load the model in the dockerized
api in linux, I ran into some problem of not being able to read Windows path in linux during the unpickling. This is
why i quickly set up a second docker compose file for training to be able to train the model within docker.

**_NOTE:_** The deployable service is only tested on a Windows 10 Machine running WSL2 Ubuntu. For the given messages
in the README.md it should return the model accuracy for the status and `POLITICS` and `WELLNESS` for the prediction
messages.

