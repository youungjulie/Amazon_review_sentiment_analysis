import pandas as pd
from sklearn.model_selection import train_test_split
from module import read_data, create_df, create_df_unlabeled

categories = ['books', 'dvd', 'electronics', 'kitchen']

# Read data from 4 categories
# books
books_pos = read_data(open('data/processed_acl/books/positive.review', encoding='UTF-8'))
books_neg = read_data(open('data/processed_acl/books/negative.review', encoding='UTF-8'))
books_labeled = create_df(books_pos, books_neg)
books_unlabeled = read_data(open('data/processed_acl/books/unlabeled.review', encoding='UTF-8'))
# dvd
dvd_pos = read_data(open('data/processed_acl/dvd/positive.review', encoding='UTF-8'))
dvd_neg = read_data(open('data/processed_acl/dvd/negative.review', encoding='UTF-8'))
dvd_labeled = create_df(dvd_pos, dvd_neg)
dvd_unlabeled = read_data(open('data/processed_acl/dvd/unlabeled.review', encoding='UTF-8'))
# electronics
elect_pos = read_data(open('data/processed_acl/electronics/positive.review', encoding='UTF-8'))
elect_neg = read_data(open('data/processed_acl/electronics/negative.review', encoding='UTF-8'))
elect_labeled = create_df(elect_pos, elect_neg)
elect_unlabeled = read_data(open('data/processed_acl/electronics/unlabeled.review', encoding='UTF-8'))
# kitchen
kit_pos = read_data(open('data/processed_acl/kitchen/positive.review', encoding='UTF-8'))
kit_neg = read_data(open('data/processed_acl/kitchen/negative.review', encoding='UTF-8'))
kit_labeled = create_df(kit_pos, kit_neg)
kit_unlabeled = read_data(open('data/processed_acl/kitchen/unlabeled.review', encoding='UTF-8'))

data_struct = pd.DataFrame(columns=['positive', 'negative', 'labeled', 'unlabeled'])
data_struct.loc['books'] = [len(books_pos), len(books_neg), len(books_labeled), len(books_unlabeled)]
data_struct.loc['dvd'] = [len(dvd_pos), len(dvd_neg), len(dvd_labeled), len(dvd_unlabeled)]
data_struct.loc['electronics'] = [len(elect_pos), len(elect_neg), len(elect_labeled), len(elect_unlabeled)]
data_struct.loc['kitchen'] = [len(kit_pos), len(kit_neg), len(kit_labeled), len(kit_unlabeled)]
print(data_struct)

books_unlabeled_df = create_df_unlabeled(books_unlabeled)
dvd_unlabeled_df = create_df_unlabeled(dvd_unlabeled)
elect_unlabeled_df = create_df_unlabeled(elect_unlabeled)
kit_unlabeled_df = create_df_unlabeled(kit_unlabeled)


#### For supervised learning ####
supervised_dataset = pd.concat([books_labeled, dvd_labeled, elect_labeled, kit_labeled]).reset_index(drop=True)
print("Number of samples in Supervised Dataset: ", len(supervised_dataset))

# train test split
X_train_temp, X_test, y_train_temp, y_test = train_test_split(supervised_dataset['sentence'],
                                                              supervised_dataset['label'], test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.25, random_state=42)

# save supervised dataset
train_df = pd.DataFrame({'sentence': X_train, 'label': y_train})
val_df = pd.DataFrame({'sentence': X_val, 'label': y_val})
test_df = pd.DataFrame({'sentence': X_test, 'label': y_test})
train_df.to_csv('data/sl_train_data.csv', index=False)
val_df.to_csv('data/sl_val_data.csv', index=False)
test_df.to_csv('data/sl_test_data.csv', index=False)

# save unlabeled data
unlabeled_df = pd.concat([books_unlabeled_df, dvd_unlabeled_df, elect_unlabeled_df, kit_unlabeled_df]).reset_index(drop=True)
unlabeled_df.to_csv('data/unlabeled_data.csv', index=False)


#### For transfer learning ####
# source dataset
source_dataset = books_labeled
print("Number of samples in Source Dataset: ", len(source_dataset))

# target dataset
target_dataset = dvd_labeled
target_dataset = target_dataset.sample(n=int(source_dataset.shape[0] * 0.1), random_state=42)
print("Number of samples in Target Dataset: ", len(target_dataset))

# save source dataset
source_df = pd.DataFrame({'sentence': source_dataset['sentence'], 'label': source_dataset['label']})
source_df.to_csv('data/source_labeled_data.csv', index=False)

# save target dataset
target_df = pd.DataFrame({'sentence': target_dataset['sentence'], 'label': target_dataset['label']})
target_df.to_csv('data/target_labeled_data.csv', index=False)

# save unlabeled data separately for source and target
source_unlabeled_df = books_unlabeled_df
target_unlabeled_df = dvd_unlabeled_df.sample(n=int(source_unlabeled_df.shape[0] * 0.1), random_state=42)
source_unlabeled_df.to_csv('data/source_unlabeled_data.csv', index=False)
target_unlabeled_df.to_csv('data/target_unlabeled_data.csv', index=False)
