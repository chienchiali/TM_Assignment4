
# coding: utf-8

# # NewTalk News Crawl

# This notebook demostrate how to crawl data from Yahoo News (https://tw.news.yahoo.com/).
# 
# However, it's incomplete. You'll have to use some advanced techniques to collect enough data for document classification.

# In[1]:

import requests
from bs4 import BeautifulSoup


# In[2]:

req = requests.get('https://newtalk.tw')
req.encoding='utf-8'


# In[3]:

soup = BeautifulSoup(req.text, 'lxml')


# ### Get categories of NewTalks

# In[90]:

categories = []
for category in soup.select('.news-category-item'):
    
    print (category.a.attrs['href'])
    #print (soup)
    categories.append(category.a.attrs['href'])
    #print(category.attrs['href'])
len(categories)


# remove unwanted categories

# In[91]:

categories = categories[0:3]
categories


# #### 先跑兩個 def

# In[104]:

def get_news_links(category):
    for i in range(1,2):
        
        #取range的頁數
        page_url = '{}{}{}'.format(category,"/",i)
        req = requests.get(page_url)
        soup = BeautifulSoup(req.text, 'lxml')

    return {article.attrs['href'] for article in soup.select(".news_title a")}


# In[85]:

def get_new_content(url):
    content = []
    req = requests.get(url)
    req.encoding='utf-8'
    soup = BeautifulSoup(req.text, 'lxml')
    content.append(soup.select_one('div.content_title h1').text)
    content.append(soup.select_one('div.mainpic_text').text)
    for p in soup.select('div.fontsize.news-content p'):
        content.append(p.text)
    return '\n'.join(content)


# ### call func 跑迴圈換頁，印出content

# In[113]:

#換頁後得到各篇文章的url
links = {}
article = []
for category in categories:
    links[category] = get_news_links(category)

for category in categories:
    for url in links[category]:
        article.append({'category':category, 'article':get_new_content(url)})


# # 製造斷詞後的檔案

# ##### 以下這段不用RUN!

# In[ ]:

from sklearn.datasets import load_files
news_data_folder = 'yahoonews_test'
# 手動把資料依照類別分別讀進來
dataset = load_files(news_data_folder,categories='world', shuffle=False)
dataset.target_names


# In[ ]:

len(dataset.data)
#一個data是一篇文章，檢查有沒有存對


# ##### Jieba 斷詞。 
# 
# 對，以下這段也不用RUN。

# In[61]:

import jieba
jieba.set_dictionary('dict.txt.big.txt')


# In[62]:

import nltk
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))


# In[63]:

import string
# add more
stops.update(string.ascii_letters + string.punctuation + string.digits)
stops.update()
# and more self-defined stopwords
stops.update(map(lambda x: x.strip().split()[0], open('stopwords (1).txt')))


# In[64]:

import io
i=0
news_words = []
for article in dataset.data:
    
    #依照各類別，把斷詞完的檔案寫進資料夾            
    out_filename = 'yahoonews_test_only3/world/world_'
    words = []
    keywords = []
    words.extend(list(jieba.cut(article)))
    keywords = [word for word in words if word not in stops]
    #print(keywords)
    news_words.append(' '.join(keywords))

    with io.open(out_filename+str(i), 'w', encoding = 'utf-8') as f:
        print(news_words[i], file = f)
        i=i+1
    
#dataset.data = news_words


# # Text Classification with SVM

# In[9]:

from sklearn.datasets import load_files
dataset = load_files('newtalks_train', encoding='utf-8')
dataset.target_names


# In[10]:

len(dataset.data)
#一個data是一篇文章
print (dataset.data[0])


# # Extracting features from text files

# ### Tokenizing text with scikit-learn

# In[30]:

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(dataset.data)
X_train_counts.shape


# In[49]:

count_vect.vocabulary_.get("法國")


# In[32]:

ngram_count_vect = CountVectorizer(ngram_range=(1, 5))
XX_train_counts = ngram_count_vect.fit_transform(dataset.data)
XX_train_counts.shape


# In[39]:

ngram_count_vect.vocabulary_.get("稍晚 宣稱")


# ### From occurrences to frequencies

# In[51]:

#沒有 ngram 的 tfidf
from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tf.shape


# In[52]:

#沒有 ngram 的 tfidf
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape


# In[67]:

#有 ngram 的 tf
from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(XX_train_counts)
X_train_tf = tf_transformer.transform(XX_train_counts)
X_train_tf.shape


# In[68]:

#有 ngram 的 tfidf
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(XX_train_counts)
X_train_tfidf.shape


# ### TfidfVectorizer

# In[69]:

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(dataset.data)
X_train_tfidf.shape


# Split data into train data and test data

# In[132]:

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.25, random_state=None)


# # Building a pipeline

# In order to make the vectorizer => transformer => classifier easier to work with, scikit-learn provides a Pipeline class that behaves like a compound classifier

# TASK: Build a vectorizer / classifier pipeline that filters out tokens that are too rare or too frequent

# In[55]:

from sklearn.pipeline import Pipeline
#SVM 
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.linear_model import SGDClassifier


# In[70]:

text_clf = Pipeline([('vect', TfidfVectorizer()),
                     ('clf', LinearSVC()),
])


# In[71]:

text_clf.fit(dataset.data, dataset.target)


# ## load test data

# In[58]:

yahoo_test = load_files('yahoonews_test_only3', encoding='utf-8')


# In[60]:

predicted = text_clf.predict(yahoo_test.data)


# In[62]:

import numpy as np
np.mean(predicted == yahoo_test.target)


# In[72]:

from sklearn import metrics
print(metrics.classification_report(yahoo_test.target, predicted,
    target_names=yahoo_test.target_names))


# ### Parameter tuning using grid search

# In[73]:

from sklearn.model_selection import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'vect__use_idf': (True, False),
              'clf__C': (1.0, 0.1, 1e-2, 1e-3),
}


# In[74]:

gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)


# In[75]:

gs_clf = gs_clf.fit(dataset.data, dataset.target)


# In[76]:

dataset.target_names[gs_clf.predict(['安全'])[0]]


# In[77]:

gs_clf.best_score_


# In[78]:

for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))


# In[79]:

clf = gs_clf.best_estimator_


# In[80]:

predicted = clf.predict(yahoo_test.data)


# In[81]:

import numpy as np
np.mean(predicted == yahoo_test.target)  


# In[82]:

from sklearn import metrics
print(metrics.classification_report(yahoo_test.target, predicted,
    target_names=yahoo_test.target_names))


# In[151]:

gs_clf.cv_results_


# In[45]:

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img=mpimg.imread('stop.png')
imgplot = plt.imshow(img)
plt.axis('off')
plt.show()


# # 以下為草稿禁區，不用看！！！

# ### Get news list

# In[6]:

req = requests.get('{}'.format(categories[0]))
soup = BeautifulSoup(req.text, 'lxml')


# In[101]:

from urllib.parse import urljoin
urls = set()
for article in soup.select(".news_title a"):
    url = article.attrs['href']
    urls.add(url)
    #print(url)
    
len (urls)


# #### Pack it into a function

# In[104]:

def get_news_links(category):
    for i in range(1,2):
        
        #取range的頁數
        page_url = '{}{}{}'.format(category,"/",i)
        req = requests.get(page_url)
        soup = BeautifulSoup(req.text, 'lxml')

    return {article.attrs['href'] for article in soup.select(".news_title a")}


# ### Get links from every category

# In[105]:

links = {}
for category in categories:
    links[category] = get_news_links(category)


# In[ ]:

links


# ## Get news content

# get the html document

# In[79]:

req = requests.get('https://newtalk.tw/news/view/2017-05-16/86842')
req.encoding='utf-8'
soup = BeautifulSoup(req.text, 'lxml')


# In[ ]:

print (soup)


# In[83]:

soup.select_one('div.content_title h1').text


# In[ ]:

for p in soup.select('div.fontsize.news-content p'):
    print(p.text)


# Pack it into a function

# In[85]:

def get_new_content(url):
    content = []
    req = requests.get(url)
    req.encoding='utf-8'
    soup = BeautifulSoup(req.text, 'lxml')
    content.append(soup.select_one('div.content_title h1').text)
    content.append(soup.select_one('div.mainpic_text').text)
    for p in soup.select('div.fontsize.news-content p'):
        content.append(p.text)
    return '\n'.join(content)


# In[86]:

url = 'https://newtalk.tw/news/view/2017-05-16/86844'
print(get_new_content(url))


# news id parse function

# In[11]:

import re
url_re = re.compile('-(\d{7,10})')

def parse_nid(url):
    m = url_re.search(url)
    return m.group(1)


# In[63]:

parse_nid(url)


# In[14]:

import os
for c in categories:
    os.mkdir(c)


# In[ ]:

for category, urls in links.items():
    for url in urls:
        print(url)
        nid = parse_nid(url)
        filename = '{0}/{1}'.format(category, nid)
        
        if os.path.isfile(filename):
            continue

        content = get_new_content(url)
        with open(filename, 'w') as f:
            print(content, file=f)


# ## Use selenium

# In[29]:

get_ipython().system('pip install selenium')


# In[30]:

from selenium import webdriver
driver = webdriver.Chrome()


# In[31]:

cat = categories[0]
driver.get('{}'.format(categories[0]))
cat


# In[18]:

urls = set()
for article in driver.find_elements_by_css_selector('.Cf a'):
    if article.text:
#         urls.add(article.text)
        urls.add(article.get_attribute('href'))
print(len(urls))


# In[19]:

# used for: https://tw.news.yahoo.com/politics
for article in driver.find_elements_by_css_selector('div#mrt-node-Col1-1-Hero a'):
    if article.text:
#         urls.add(article.text)
        urls.add(article.get_attribute('href'))
print(len(urls))


# In[ ]:




# In[204]:

import time
def get_urls(cat):
    driver.get('{}'.format(categories[0]))

    urls = set()
    for article in driver.find_elements_by_css_selector('.js-stream-content > div:not(.controller) a:not(.comment-btn-link)'):
        if article.text:
            urls.add(article.get_attribute('href'))
    for article in driver.find_elements_by_css_selector('div#mrt-node-Col1-1-Hero a'):
        if article.text:
            urls.add(article.get_attribute('href'))
    print(len(urls))
    return urls


# In[189]:

len(links['politics'])


# In[ ]:



