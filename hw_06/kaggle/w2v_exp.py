from gensim.models.word2vec import Word2Vec
from bs4 import BeautifulSoup
import numpy as np
import gensim, logging
from gensim.models import KeyedVectors


model = gensim.models.Word2Vec.load("trained.vec")