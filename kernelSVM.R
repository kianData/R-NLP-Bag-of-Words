# Natural Language Processing

# Importing the dataset
dataset_original = read.delim('Restaurant_Reviews.tsv', quote = '', stringsAsFactors = FALSE)

# Cleaning the texts
# install.packages('tm')
# install.packages('SnowballC')
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset_original$Review))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
# needs SnowballC:
corpus = tm_map(corpus, removeWords, stopwords())
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)

# Creating the Bag of Words model
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999)
dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_original$Liked

# Importing the dataset
# dataset = read.csv('Social_Network_Ads.csv')
# dataset = dataset[3:5]

# Encoding the target feature as factor
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Kernel SVM to the Training set
# install.packages('e1071')
library(e1071)
classifier = svm(formula = Liked ~ .,
                 data = training_set,
                 type = 'nu-classification',
                 kernel = 'radial')

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692])

# Making the Confusion Matrix
cm = table(test_set[, 692], y_pred)

# Evaluating Performance
tn = cm[1,1]
tp = cm[2,2]
fp = cm[2,1]
fn = cm[1,2]

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1Score = 2 * precision * recall / (precision + recall)

performance <- matrix(, nrow = 2, ncol = 4)
performance[1, 1] = 'accuracy'
performance[1, 2] = 'precision'
performance[1, 3] = 'recall'
performance[1, 4] = 'f1Score'
performance[2, 1] = accuracy
performance[2, 2] = precision
performance[2, 3] = recall
performance[2, 4] = f1Score

rm(tp, tn, fp, fn, accuracy, precision, recall, f1Score)