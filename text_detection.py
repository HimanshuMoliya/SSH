#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag


# In[19]:


nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


# In[20]:


ex = 'European authorities fined Google a record $5.1 billion on 22/02/2019 for abusing its power in the mobile phone market and ordered the company to alter its practices'


# In[21]:


def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent


# In[22]:


sent = preprocess(ex)


# In[23]:


sent


# In[24]:


pattern = 'NP: {<DT>?<JJ>*<NN>}'


# In[25]:


cp = nltk.RegexpParser(pattern)
cs = cp.parse(sent)
print(cs)


# In[26]:


from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint
iob_tagged = tree2conlltags(cs)
pprint(iob_tagged)


# In[27]:


ne_tree = nltk.ne_chunk(pos_tag(word_tokenize(ex)))
print(ne_tree)


# In[28]:


import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()


# In[29]:


doc = nlp('European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices')
pprint([(X.text, X.label_) for X in doc.ents])


# In[30]:


pprint([(X, X.ent_iob_, X.ent_type_) for X in doc])


# "B" means the token begins an entity, "I" means it is inside an entity, "O" means it is outside an entity, and "" means no entity tag is set.
