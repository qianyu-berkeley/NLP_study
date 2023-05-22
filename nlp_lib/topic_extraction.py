from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

def LDA(df, col):

    cv = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = cv.fit_transform(df[col])

    LDA = LatentDirichletAllocation(n_components=7,random_state=42)
    LDA.fit(dtm)
    
    for index, topic in enumerate(LDA.components_):
        print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
        print([cv.get_feature_names()[i] for i in topic.argsort()[-15:]])
        print('\n')

    topic_results = LDA.transform(dtm)
    df['topic'] = topic_results.argmax(axis=1)
    return df

def NMF(df, col):

    tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = tfidf.fit_transform(df[col])

    nmf_model = NMF(n_components=7,random_state=42)
    nmf_model.fit(dtm)

    for index,topic in enumerate(nmf_model.components_):
        print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
        print([tfidf.get_feature_names()[i] for i in topic.argsort()[-15:]])
        print('\n')

    topic_results = nmf_model.transform(dtm)
    df['topic'] = topic_results.argmax(axis=1)
    return df