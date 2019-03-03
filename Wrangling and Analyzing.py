#!/usr/bin/env python
# coding: utf-8

# In[2]:


import datetime as dt
import json
import numpy as np
import pandas as pd
import re
import requests
import tweepy


# # Gathering the data

# In[3]:


archive = pd.read_csv("twitter-archive-enhanced.csv")
archive.set_index("tweet_id", inplace = True)
archive.head(2)


# In[4]:


tsv_url = "https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv"
r = requests.get(tsv_url)

with open(tsv_url.split('/')[-1], mode = 'wb') as file:
    file.write(r.content)
    
images = pd.read_csv('image-predictions.tsv', sep = '\t')
images.head(2)


# In[5]:


consumer_key = ""
consumer_secret = ""
access_token = ""
access_token_secret = ""

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, parser=tweepy.parsers.JSONParser(), wait_on_rate_limit=True)


# In[6]:


df = []
exceptions = []
tweet_id = images['tweet_id']

for id in tweet_id:
    try:
        page = api.get_status(id)
        favorites = page['favorite_count']
        retweets = page['retweet_count']
        time = pd.to_datetime(page['created_at'])
        df.append({'tweet_id': int(id),
                        'favorites': int(favorites),
                        'retweets': int(retweets)})
    
    except Exception as e:
        exceptions.append(id)


# In[7]:


exceptions


# In[8]:


exceptions2 = []
for e in exceptions:
    try:
        page = api.get_status(e)
        favorites = page['favorite_count']
        retweets = page['retweet_count']
        time = pd.to_datetime(page['created_at'])
        df.append({'tweet_id': int(e),
                        'favorites': int(favorites),
                        'retweets': int(retweets)})
        
    except Exception:
        exceptions2.append(id)


# In[ ]:


df = pd.DataFrame(df, columns = ['tweet_id', 'favorites', 'retweets'])
df.to_csv('tweet_json.txt', encoding = 'utf-8')


# In[ ]:


df = pd.read_csv('tweet_json.txt', encoding = 'utf-8')
df.set_index('tweet_id', inplace = True)
df.tail()


# In[ ]:


images.set_index('tweet_id', inplace = True)
df2 = pd.merge(left=archive, right=images, left_index=True, right_index=True, how='left')
df2 = pd.merge(left=df2, right=df, left_index=True, right_index=True, how='left')
df2.to_csv('df2copy.csv', encoding = 'utf-8')


# # Assessing the Data

# In[ ]:


archive.info()


# In[ ]:


archive.name.value_counts()


# In[ ]:


archive.rating_denominator.value_counts()


# In[ ]:


archive.rating_numerator.value_counts()


# In[ ]:


archive.head()


# In[ ]:


images.info()


# In[ ]:


tweet = pd.read_csv("tweet_json.txt", encoding = 'utf-8')
tweet.info()


# ### Judging the quality
# - Columns are having empty values, like *in_reply_to_status*, *in_reply_to_user_id*, *retweeted_status_id*, *retweeted_status_user_id*, *retweeted_status_timestamp*.
# - The *name* column has many entries which do not look like names. The most frequent entry in name column is "a", which is not a name.
# - Unususal values are there in Numerator as well as Denominator.
# - The timestamp column is an object. It has to be a datetime object.
# - There are 2075 rows in the images dataframe and 2356 rows in the archive dataframe.
# - In several columns, null values are not treated as null values.
# 
# ### Checking the tidiness
# 
# - The dog stages have values as columns, instead of one column filled with their values.
# - We don't need the *Unnamed: 0* column from the *tweet* dataframe.
# - The columns for dog breed predictions can be condensed.

# In[ ]:


df = pd.read_csv("df2copy.csv")


# # Cleaning the extras
# ### Define:
# ### Deleting an extra column
# ### Code area:

# In[ ]:


del(df['Unnamed: 0'])


# ### Test casing:

# In[ ]:


df.columns


# ### Defining extra terms:
# ### Convert timestamp to datetime object
# ### Coding area:

# In[ ]:


df['timestamp'] = pd.to_datetime(df['timestamp'])


# In[ ]:


df.info()


# ### Define: 
# 
# ### Remove Retweets and Tweets which does not include image
# 
# ### Code:

# In[ ]:


# removing the tweets without images
df = df[pd.notnull(df['jpg_url'])]


# In[ ]:


# removing retweets
df = df[pd.isnull(df['retweeted_status_id'])]
df.shape[0]


# In[ ]:


df.drop(['retweeted_status_id', 'retweeted_status_user_id', 'retweeted_status_timestamp'], axis = 1, inplace = True)


# ### Test:

# In[ ]:


df.columns


# ### Define:
# 
# ### Condensing Dog Type columns
# 
# ### Code:

# In[ ]:


dog_type = []

x = ['pupper', 'puppo', 'doggo', 'floof']
y = ['pupper', 'puppo', 'doggo', 'floof']

for row in df['text']:
    row = row.lower()
    for word in x:
        if word in str(row):
            dog_type.append(y[x.index(word)])
            break
    else:
        dog_type.append('None')

df['dog_type'] = dog_type


# In[ ]:


df['dog_type'].value_counts()


# In[ ]:


# removing extra columns
df.drop(['doggo', 'floofer', 'pupper', 'puppo'], axis=1, inplace=True)


# ### Test:

# In[ ]:


#test
df.columns


# ### Define:
# ### Condensing dog breed predictions
# ### Coding area:

# In[ ]:


breed = []
conf= []

def breed_conf(row):
    if row['p1_dog']:
        breed.append(row['p1'])
        conf.append(row['p1_conf'])
    elif row['p2_dog']:
        breed.append(row['p2'])
        conf.append(row['p2_conf'])
    elif row['p3_dog']:
        breed.append(row['p3'])
        conf.append(row['p3_conf'])
    else:
        breed.append('Unidentifiable')
        conf.append(0)

df.apply(breed_conf, axis = 1)

df['breed'] = breed
df['confidence'] = conf


# In[ ]:


#removing the processed columns
df.drop(['p1', 'p1_conf', 'p1_dog', 'p2', 'p2_conf', 'p2_dog', 'p3', 'p3_conf', 'p3_dog',], axis=1, inplace=True)


# ### Test:

# In[ ]:


df.head(2)


# ### Define:
# ### Removing the useless columns
# ### Coding area:

# In[ ]:


df['in_reply_to_status_id'].value_counts()


# In[ ]:


df['in_reply_to_user_id'].value_counts()


# These all reply to a single user id, i.e., @dog_rates

# In[ ]:


df.drop(['in_reply_to_status_id', 'in_reply_to_user_id'], axis=1, inplace=True)


# ### Test:

# In[ ]:


df.columns


# ### Define:
# ### Extract Dog Rates and Dog Count
# ### Code:

# In[ ]:


rates = []

#raw_rates = lambda x: rates.append(re.findall(r'(\d+(\.\d+)|(\d+))\/(\d+0)', x, flags=0))

df['text'].apply(lambda x: rates.append(re.findall(r'(\d+(\.\d+)|(\d+))\/(\d+0)', x, flags=0)))

rating = []
dog_count = []

for item in rates:
    
    # for tweets with no rating, but a picture, so a dog_count of 1
    if len(item) == 0:
        rating.append('NaN')
        dog_count.append(1)
        
    # for tweets with single rating and dog_count of 1
    elif len(item) == 1 and item[0][-1] == '10':
        rating.append(float(item[0][0]))
        dog_count.append(1)
   
    # for multiple ratings
    elif len(item) == 1: 
        a = float(item[0][0]) / (float(item[0][-1]) / 10) 
        rating.append(a)
        dog_count.append(float(item[0][-1]) / 10)
   
    # for tweets with more than one rating
    elif len(item) > 1: 
        total = 0
        r = []
        for i in range(len(item)):
            if item[i][-1] == '10': #one tweet has the phrase '50/50' so I'm coding to exclude it
                r.append(item[i])
        for rate in r:
            total = total + float(rate[0])
        a = total / len(item)
        rating.append(a)
        dog_count.append(len(item))
   
    # if any error has occurred
    else:
        rating.append('Not parsed')
        dog_count.append('Not parsed') 
        
df['rating'] = rating # not need to also add denominator since they are all 10!
df['dog_count'] = dog_count
df['rating'].value_counts()


# In[ ]:


df.drop(['rating_numerator', 'rating_denominator'], axis=1, inplace=True)


# ### Test:

# In[ ]:


df.info()


# In[ ]:


df['dog_count'].value_counts()


# ### Define:
# ### Extract Names
# ### Code:

# In[ ]:


df['text_split'] = df['text'].str.split()


# In[ ]:


names = []

# use string starts with method to clean this up

def extract_names(row):
    
    # 'named Phineas'           
    if 'named' in row['text'] and re.match(r'[A-Z].*', row['text_split'][(row['text_split'].index('named') + 1)]): 
            names.append(row['text_split'][(row['text_split'].index('named') + 1)])
    
    # 'Here we have Phineas'
    elif row['text'].startswith('Here we have ') and re.match(r'[A-Z].*', row['text_split'][3]):
            names.append(row['text_split'][3].strip('.').strip(','))
            
    # 'This is Phineas'
    elif row['text'].startswith('This is ') and re.match(r'[A-Z].*', row['text_split'][2]):
            names.append(row['text_split'][2].strip('.').strip(','))
    
    # 'Say hello to Phineas'
    elif row['text'].startswith('Say hello to ') and re.match(r'[A-Z].*', row['text_split'][3]):
            names.append(row['text_split'][3].strip('.').strip(','))
    
    # 'Meet Phineas'
    elif row['text'].startswith('Meet ') and re.match(r'[A-Z].*', row['text_split'][1]):
            names.append(row['text_split'][1].strip('.').strip(','))
            
    else:
        names.append('Nameless')
        
        
df.apply(extract_names, axis=1)

df['names'] = names


# In[ ]:


df['names'].value_counts()


# "a", "the" and all non-name words have been removed.

# In[ ]:


df.drop(['text_split'], axis=1, inplace=True)


# In[ ]:


df.loc[df['names'] == 'Nameless', 'names'] = None
df.loc[df['breed'] == 'Unidentifiable', 'breed'] = None
df.loc[df['dog_type'] == 'None', 'dog_type'] = None
df.loc[df['rating'] == 0.0, 'rating'] = np.nan
df.loc[df['confidence'] == 0.0, 'confidence'] = np.nan


# ### Test:

# In[ ]:


df.info()


# Saving the cleaned file.

# In[ ]:


df.to_csv('twitter_archive_master.csv', encoding = 'utf-8')


# # Analyzing the Data

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
#import matplotlib
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('twitter_archive_master.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)


# ### Retweets, Favorites and Ratings Correlation

# In[ ]:


df[['favorites', 'retweets']].plot(style = '.', alpha = 0.4)
plt.title('Favorites and Retweets with Time')
plt.xlabel('Date')
plt.ylabel('Count');


# In[ ]:


df.plot(y ='rating', ylim=[0,14], style = '.', alpha = 0.4)
plt.title('Rating with Time')
plt.xlabel('Date')
plt.ylabel('Rating');


# Here you can see the gradual increase of both favorites and retweets over time.

# In[ ]:


df[['favorites', 'rating', 'retweets']].corr(method='pearson')


# So Brant was right, there are more ratings above 10. Still don't know the reason why there are so much high ratings.
# 
# So let's see if dogs with higher ratings were getting more favorites and retweets. According to me, if the dogs are getting better they should be getting more favorites and retweets along with the higher rating. There is a strong correlation between favorites and retweets. This means that if the tweet is good in general then there will be more retweets and favorites.
# 
# Yet there is no correlation between rating and retweets or rating and favorites. It can be because the dogs are not actually getting better. It can be that 'lower quality' dogs are given funnier captions. In this case, it is the caption that is getting more retweets and favorites, rather than the dog itself.

# ### Dog Stages Stats

# In[ ]:


df.boxplot(column='rating', by='dog_type');


# In[ ]:


df.groupby('dog_type')['rating'].describe()


# In[ ]:


df.reset_index(inplace=True)
df.groupby('dog_type')['timestamp'].describe()


# So puppers are getting much lower rates than the other dog types. They have several low outliers which decrease the mean to 10.6.
# 
# Floofers are consistently rated above 10. I don't know whether they are really good or the rating just gets higher with time. Maybe we can see if 'floof' is a newer term.
# 
# Here we see that 'floof' is not a new term, first seen on January 2016. So we can say that floofer are consistently good dogs.

# ### Most Rated Breeds

# In[ ]:


top=df.groupby('breed').filter(lambda x: len(x) >= 20)
top['breed'].value_counts().plot(kind = 'bar')
plt.title('The Most Rated Breeds');


# It's difficult to know why these breeds are the top breeds. It could be because they are commonly owned. Or they could be the easiest to identify by the AI that identified them.

# In[ ]:


top.groupby('breed')['rating'].describe()


# In[ ]:


df['rating'].describe()


# In[ ]:


df[df['rating'] <= 14]['rating'].describe()


# Here we have a statistical comparison of the top breeds with all the ratings. Only one of the top breeds has a mean higher than the total population mean. This is because of these two ratings: 420 and 1776.

# Excluding outliers bring down the mean to 10.55.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




