# Movie Review Sentiment Analysis
---
## About

This is a mini project for SC1015 - Introduction to Data Science and Artificial Intelligence. Our project makes use of supervised machine learning using natural language processing (NLP) techniques to predict the binary sentiments of movie reviews written by anyone.

---
## Analysis with Data Science Pipeline

Our data is collected from a repository of IMDB movie reviews. Our dataset was obtained from Kaggle, [IMDB 50K Movie Reviews (TEST your BERT)](https://www.kaggle.com/datasets/atulanandjha/imdb-50k-movie-reviews-test-your-bert).

~>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>~
<br> In reality, movie reviews contain valuable information about the quality and appeal of a movie, but reading and analysing each review manually can be time-consuming and subjective. Our project aims to save time, provide objective insights and help individuals and businesses make more informed decisions about which movies to watch or invest in.
<br> People who are looking for guidance in choosing a movie to watch need to know how well-received a particular movie is. With our sentiment analysis, the lengthy reviews can be categorised to "positive" and "negative" reviews for them to understand the distribution of sentiments.
~>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>~
<br> To prepare the data for the sentiment analysis, we cleaned the formatting of the data (movie reviews), by applying [stemming](https://www.statistics.com/glossary/stemming/#:~:text=In%20processing%20unstructured%20text%2C%20stemming,the%20single%20stem%20%22process.%22) on our dataset. We then make use of histograms and bar graphs to categorise the datasets into positive and negative reviews to study them. By using a series of data visualisation tools like word clouds, we extracted the frequency of common words within the categories, and analysed them.
<br> Thereafter, we recognise the significant patterns and optimise the algorithm by splitting the dataset into 'test' and 'train' sets, then applying the models to both sets and analyse. Then we use [tokenization](https://www.statistics.com/glossary/tokenization/#:~:text=Tokenization%3A,can%20also%20count%20as%20tokens) on our data, followed by vectorisation using [TFIDF (term frequency inverse document frequency)](https://tinyurl.com/598bynkc).

~>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>~

We used the following machine learning models:
<br> Logistic Regression Model, Multinomial Naive Bayes Model, Linear Support Classification Model, Decision Tree.

The models were then evaluated using a [Confusion Matrix](https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62) and assessed based on their [ROC AUC Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html). With the selection of the most optimal model, the detection of sentiment was correct to a high degree of accuracy.
<br> Gone are the days where one has to go through multiple pages of movie reviews in order to decide whether the overal sentiment is good or bad. This allows individuals and business make more better and more-informed decisions, especially when dealing with large datasets.

~>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>~
<br> _No known legal boundaries were violated during our analysis. However, our movie review sentiment analysis might rob away the organic touch to watching movies and might even provide several spoilers to movie-goers._


---

## Problem Definition
1. Can we predict the general sentiment of a movie review?
2. Which model is the most suitable to predict the sentiment of a movie review?

---

## Models Used
*   Logistic Regression Model
*   Multinomial Naive Bayes Model
*   Linear Support Classification Model
*   Decision Tree

---

## Conclusion
1. We can successfully predict the general sentiment of a movie review.
2. We initially thought that the Logistic Regression Model is the most suitable to predict the sentiment of a movie review, but later realised that the Linear Support Vector Classification Model is the most suitable.

---
## What have we learnt from the project?
1. Text data cleaning and stemming.
2. TFIDF Vectorization.
3. Logistic Regression model, SVC model, Multinomial Naive Bayes Model.
4. Exporting model to .joblib file for future use.
5. Flask library for GUI.
6. Python and OS communication.

---

## Key Takeaways
Although the different models that we trialed can all predict the final sentiment of a movie review, the most suitable model is the Linear Support Classification Model as it has the highest precision value among the 4 models.

---

## Contributors

*   [Timothy Chang Kai En](https://github.com/legithubble) - Initial data visualisation, Data cleaning, Multinomial Naive Byers Model, Linear Support Vector Classification Model, Script formulation, Sentiment Detector GUI
*   [Toh Si En Ernneth](https://github.com/potatohburritoh/) - Data cleaning, Word cloud, README, Script formulation, Video Slides, Video Presentation
*   [Tuan Nguyen](https://github.com/tuanisworkingonsomeprojects) - Stemming process, Logistic Regression, Decision Tree, Script formulation, Sentiment Detector (Back-end)

---

## References

*   [Kaggle](https://www.kaggle.com/datasets) (dataset)
*   [statistics.com](https://www.statistics.com/) (definitions)
*   [MonkeyLearn Blog](https://tinyurl.com/598bynkc) (understanding TFIDF)
*   [StatQuest with Josh Starmer](https://www.youtube.com/@statquest) via YouTube (understanding ML models)
*   [Deep learning on Movie Reviews Dataset (IMDB Dataset - 50k reviews) | Deep Learning Project 2](https://youtu.be/ybzeyAfWh7U) via YouTube
*   [Movies Reviews Sentiment Analysis in NLP  | Natural Language Processing #NLP](https://youtu.be/xRuy7yi2Sp8) via YouTube

---
