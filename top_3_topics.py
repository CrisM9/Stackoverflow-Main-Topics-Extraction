import pandas as pd
import re
import nltk
import wordninja
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
lemma=nltk.stem.WordNetLemmatizer()

my_data=pd.read_csv('data.csv', delimiter=',')

def text_processing(post):
    post=str(post)
    post = post.lower()
    post = re.sub("<!--?.*?-->", "", post)
    post= re.sub("(\\d|\\W)+", " ", post)
    post.split()
    post=[lemma.lemmatize(word) for word in wordninja.split(post)]
    post=" ".join(post)
    return post

my_data['processed_text']=my_data.title+my_data.body
my_data['processed_text']=my_data.processed_text.apply(text_processing)

doc=my_data['processed_text'].tolist() #lista coloana "processed_text
cv=CountVectorizer(analyzer='word', stop_words='english', max_df=0.85, ngram_range=(1,2))
word_count=cv.fit_transform(my_data['processed_text'])

tfidf_transf=TfidfTransformer()
tfidf_transf.fit(word_count)
tf_idf_matrix=tfidf_transf.transform(word_count).toarray()

dict_sent={}
for doc_nr in range (0, len(tf_idf_matrix)):
    df_tfidf=pd.DataFrame(tf_idf_matrix[doc_nr], index=cv.get_feature_names(), columns=['tf_idf_scores']).sort_values(by=['tf_idf_scores'], ascending=False).head(3)
    dict_sent[doc_nr]=df_tfidf.index.values.tolist()

df_results=pd.DataFrame(columns=['Title', 'Top 3 topics'])
df_results['Title']=my_data['title'].values
df_results['index']=df_results.reset_index().index
df_results['Top_3_topics']=df_results['index'].map(dict_sent)