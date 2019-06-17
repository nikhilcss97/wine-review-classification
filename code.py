# --------------
import pandas as pd

# Code Starts Here
def load_data(path= path):
    df= pd.read_csv(path)
    df= df[['description', 'variety']]
    df= df.iloc[:80000]
    print(df.head())
    return df

df= load_data()
# Code Ends here


# --------------
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from string import punctuation
from nltk.stem import LancasterStemmer
import numpy as np

custom = set(stopwords.words('english')+list(punctuation)) 

# Code Starts Here
df= load_data()
df= df.groupby('variety').filter(lambda x: len(x)>1000)

def to_lower(x):
    return x.lower()

df= df.apply(np.vectorize(to_lower))
df.variety= df.variety.str.replace(" ", "_")

df= df.reset_index()

all_text= pd.DataFrame(df.description)
lancaster= LancasterStemmer()

all_text_list= list(all_text.description)
stemmed_text= list()

for i in range(len(all_text_list)):
    stemmed_text.append(lancaster.stem(all_text_list[i]))

all_text= pd.DataFrame({'description':stemmed_text})
# Stemming the data

def remove_stopwords(x):
    clean= [word for word in x.split() if word not in custom]
    return " ".join(clean)

all_text= all_text.apply(np.vectorize(remove_stopwords))

# Initialize Tfidf vectorizer and LabelEncoder
tfidf= TfidfVectorizer(stop_words= 'english')
le= LabelEncoder()

tfidf.fit(all_text.description)
X= tfidf.transform(all_text.description).toarray()
y= pd.DataFrame(df.variety)

y= le.fit_transform(y.variety)

# print(type(y))
# Code Ends Here


# --------------
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

# Code Starts here

# Splitting the dataset
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= .3, random_state= 42)

# Initializing Navie bayes
nb= MultinomialNB()
nb.fit(X_train, y_train)

y_pred_nb= nb.predict(X_test)

nb_acc= accuracy_score(y_test, y_pred_nb)
# Code Ends here


# --------------
from collections import Counter
from sklearn.model_selection import train_test_split
import numpy as np

# Code Starts Here
#Load the dataset from path
df= pd.read_csv(path)

#Extract description and variety
df1= df[['description', 'variety']]

#Access top 10 categories from variety column
variety= list(df.variety)
counter= Counter(variety)

top_10_varieties= counter.most_common(10)
top_10_varieties= list(dict(top_10_varieties).keys())

#Map the top 10 varieties on df1
mapped= df1.variety.apply(lambda x: True if x in top_10_varieties else False)

description_list= list(df1.loc[mapped].description)

varietal_list= [variety for variety in df1.variety.tolist() if variety in top_10_varieties]
varietal_list= np.array(varietal_list)

count_vect= CountVectorizer()
tfidf_transformer= TfidfTransformer()

x_train_counts= count_vect.fit_transform(description_list)

x_train_tfidf= tfidf_transformer.fit_transform(x_train_counts)

# print(type(x_train_tfidf), type(varietal_list))
# print(x_train_tfidf.shape, len(varietal_list))

train_x, test_x, train_y, test_y= train_test_split(x_train_tfidf,varietal_list, test_size= .3, random_state= 42)

clf= MultinomialNB()

clf.fit(train_x, train_y)

y_score= clf.predict(test_x)

nb_cv_acc= accuracy_score(test_y, y_score)

# Code Ends Here


# --------------
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")


# Code Starts Here
rfc= RandomForestClassifier()

rfc.fit(train_x, train_y)
y_pred_rf= rfc.predict(test_x)

random_forest_acc= accuracy_score(test_y, y_pred_rf)
# Code Ends Here



# --------------
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression

clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial',max_iter = 500,random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = MultinomialNB()


#Creation of list of models
Model_List=[('Logistic Regression', clf1),
            ('Random Forest'      , clf2),
            ('MultinomialNB'      , clf3)]

eclf1 = VotingClassifier(estimators=Model_List, voting='hard')
eclf1 = eclf1.fit(train_x, train_y)
hard_acc1 = eclf1.score(test_x,test_y)
print("Accuracy Score hard: ",hard_acc1)


eclf2 = VotingClassifier(estimators=Model_List, voting='soft')
eclf2 = eclf2.fit(train_x, train_y)
soft_acc1 = eclf2.score(test_x,test_y)
print("Accuracy Score soft: ",soft_acc1)


