import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import joblib
from wordcloud import WordCloud

# plot the word clod based on the feature importance

# top 20 important features
# train data
train_data = pd.read_csv('data/sl_train_data.csv')
X_train = train_data['sentence']
y_train = train_data['label']

# validation data
val_data = pd.read_csv('data/sl_val_data.csv')
X_val = val_data['sentence']
y_val = val_data['label']

# test data
test_data = pd.read_csv('data/sl_test_data.csv')
X_test = test_data['sentence']
y_test = test_data['label']

All_data = pd.concat([X_train, X_val, X_test], axis=0)

# Count Vectorizer
count_vect = CountVectorizer(stop_words='english')
All_data_counts = count_vect.fit_transform(All_data)
X_train_counts = joblib.load('data/featured/X_train_counts.pkl')

# TF-IDF Vectorizer
tfidf_vect = TfidfVectorizer(stop_words='english')
All_data_tfidf = tfidf_vect.fit_transform(All_data)
X_train_tfidf = joblib.load('data/featured/X_train_tfidf.pkl')


# best model
best_model = joblib.load('best_model.pkl')
best_model.fit(X_train_counts, y_train)
importances = best_model.feature_importances_
print("Best model feature importances: ", importances)

all_features = count_vect.get_feature_names()

# sort features based on the importance
indices = np.argsort(importances)[::-1]
print("Feature ranking:")
for f in range(20):
    print("%d. feature %s (%f)" % (f + 1, all_features[indices[f]], importances[indices[f]]))

# plot the word cloud
wordcloud = WordCloud(background_color="white", max_words=20, width=800, height=400)
wordcloud.generate_from_frequencies(dict(zip(all_features, importances)))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
