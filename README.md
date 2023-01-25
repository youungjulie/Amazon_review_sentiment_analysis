# Amazon_review_sentiment_analysis

Sentiment analysis is always a trending topic in natural language processing fields. 
This project digs into this problem and predicts customers' sentiments as positive or negative 
using the "Multi-Domain Sentiment Dataset (version 2.0) ". This dataset contains amazon reviews in 4 domains: 
book, DVD, electronic, and kitchen. I used the traditional ML methods including Logistic Regression, 
Support Vector Machine, Random Forest, and XGBoost. I also developed this content into a transfer learning problem, 
and want to predict sentiments of DVD reviews as the target domain from books reviews as the source domain using the **Subspace Alignment** method 
and **TrAdaBoost** method. The best model for sentiment prediction is the **Random Forest Classifier**, 
which generated the most accurate prediction with 83.00% accuracy. The best method for the Transfer Learning model is **TrAdaBoost** which result in 76% accuracy. 

For detail report of the project, please review the file **"Final Report.pdf"**
