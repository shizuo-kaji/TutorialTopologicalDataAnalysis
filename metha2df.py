#!/usr/bin/env python
## reference: http://akumano.xyz/posts/arxiv-keyword-extraction-part1/
## Requirements
# conda install gensim nltk
# pip install kmapper
#%%
import os, gzip, glob
import pickle
from datetime import datetime
import xml.etree.ElementTree as ET
import pandas as pd
import argparse
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
#from nltk.stem.porter import PorterStemmer as st
from nltk.stem.lancaster import LancasterStemmer as st
from gensim import models
import gensim
from gensim.models.doc2vec import TaggedDocument
import numpy as np
from sklearn.preprocessing import LabelEncoder
import kmapper
from sklearn import datasets,cluster
from sklearn.manifold import Isomap
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

arXiv_prefix = '{http://arxiv.org/OAI/arXiv/}'

##
def parse_author(a):
    keyname = a.find(arXiv_prefix + "keyname")
    keyname = keyname.text if keyname is not None else ''
    forenames = a.find(arXiv_prefix + "forenames")
    forenames = forenames.text if forenames is not None else ''
    return ' '.join( (forenames, keyname) )

def metha2df(dirname):
    files = glob.glob(os.path.join(dirname,'**/*.xml.gz'), recursive=True)
    identifier, title, authors, abstract, categories, submitted, time = ([] for i in range(7))

    ## process files
    for xmlgz in files:
        print(os.path.basename(xmlgz), end=' ')
        with gzip.open(xmlgz, 'rb') as f:
            xml = ET.parse(f)
            root = xml.getroot()
            for record in root.find('ListRecords').findall("record"):
                metadata = record.find('metadata').find(arXiv_prefix + "arXiv")
                if metadata is None:
                    continue
                identifier.append(metadata.find(arXiv_prefix + "id").text)
                s = metadata.find(arXiv_prefix + "created").text
                sub = datetime.strptime(s, "%Y-%m-%d")
                submitted.append( sub )
                time.append( sub.timestamp() )

                title.append(metadata.find(arXiv_prefix + "title").text )
                abstract.append( metadata.find(arXiv_prefix + "abstract").text.strip() )

                authors_list = metadata.find(arXiv_prefix + 'authors')
                authors.append( tuple( [parse_author(a) for a in authors_list] ) )

                categories_list = metadata.find(arXiv_prefix + "categories").text
                categories.append( tuple( categories_list.split() ) )
        print(' Done.')

    ## create dataframe
    df = pd.DataFrame({
        'identifier': identifier,
        'title': title,
        'authors': authors,
        'abstract': abstract,
        'categories': categories,
        'submitted': submitted,
        'time': time
        })
    df['abstract'] = df['abstract'].apply(lambda t: t.replace('\n',' '))
    df['title'] = df['title'].apply(lambda t: t.replace('\n',' '))
    return(df)

## doc2vec
def fit_doc2vec(df):
    # parameters and preprocessing need to be tuned
    nltk.download('punkt')
    nltk.download('stopwords')
    stop_words = nltk.corpus.stopwords.words('english')
    stop_words.extend(["'", '"', ':', ';', '.', ',', '-', '!', '?', "'s","\\","$"])
    stemmer = st()
    sentences = []
    for i in range(len(df)):
    #    sentence = sent_tokenize(df['abstract'][i])
        t = '. '.join( [df['title'][i],df['abstract'][i]] )
        sentence = gensim.utils.simple_preprocess(t)
        words = []
        for s in sentence:
            for w in word_tokenize(s):
                if w.lower() not in stop_words:
                    words.append( stemmer.stem(w.lower()) )
        sentences.append( TaggedDocument(words = words, tags=[i]) )
    return(models.Doc2Vec(documents=sentences, vector_size=30, window=15, min_count=1, workers=4, dm=0))  ## parameter tuning


#%%
#### start here
# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--metha_dir', default='.metha', type=str, help='Directory containing metha files')
parser.add_argument('--df_name', '-d', default='math_2007.pkl', type=str, help='name of the dataframe file')
parser.add_argument('--n_samples', '-n', default=50000, type=int, help='the number of samples to visualise')
parser.add_argument('--model', '-m', default='doc2vec_arxiv.model', type=str, help='name of the doc2vec model')
parser.add_argument('--target', '-t', default='doc', type=str, choices=['word','doc'], help='what vectors to visualise')
parser.add_argument('--output', '-o', default='output.html', type=str, help='name of the output html file')

#args = parser.parse_args()

args = parser.parse_args('')

## Main
# parse downloaded data by metha and compile into a dataframe
if os.path.isfile(args.df_name):
    print("reading database from file: ", args.df_name)
    with open(args.df_name,'rb') as f:
        df = pickle.load(f)
else:
    print("Parsing metha files...")
    df = metha2df(args.metha_dir)
    with open(args.df_name, 'wb') as f:
        pickle.dump(df, f)
#  check the dataframe
#print(df.keys())
#print(df['abstract'][0])

# fit the doc2vec model to the dataframe
if os.path.isfile(args.model):
    print("reading the model from file: ", args.model)
    model = models.Doc2Vec.load(args.model)
else:
    print("Fitting Doc2Vec...")
    model = fit_doc2vec(df)
    model.save(args.model)

# Visualisation
print("Creating Visualisation...")
# define label and filter
n = args.n_samples
km = kmapper.KeplerMapper(verbose=2)
#f = km.project(v) 
#%% Kepler Mapper (use only 50000 samples)

if args.target == 'word':
    v = model.wv.vectors
    print(v.shape)
    projected_X = km.fit_transform(v[:n],
        projection=[PCA(n_components=2)],
        scaler=[None, None, MinMaxScaler()])
    graph = km.map(projected_X,
                   #inverse_X=None,
                   clusterer=cluster.AgglomerativeClustering(n_clusters=3,linkage="complete",affinity="cosine"),)
                   #overlap_perc=0.33)
    html = km.visualize(graph, path_html=args.output,
        custom_tooltips=np.array(model.wv.index_to_key[:n])
    )
else:
    label = [w[0] for w in df['categories']]
    le = LabelEncoder()
    le = le.fit(label)
    color = le.transform(label)
    v = np.load(args.model+".dv.vectors.npy")
    print(v.shape)
    projected_X = km.fit_transform(v[:n],
        projection=[PCA(n_components=2)],
        scaler=[None, None, MinMaxScaler()])
    ## TODO: parameter tuning
    f = df['time']  # filter is submission date
    graph = km.map(X=v[:n], lens=f[:n], 
                    #overlap_perc=0.50,
                    clusterer=cluster.DBSCAN(eps=0.2, min_samples=5, metric="cosine"))
    #                clusterer=cluster.AgglomerativeClustering(n_clusters=5,linkage="complete",affinity="cosine"))
    html = km.visualize(graph, path_html=args.output,
        #color_values=color, 
        #custom_tooltips=df['categories']
        custom_tooltips=df['title']
    )

#import networkx as nx
#nx_graph = kmapper.adapter.to_nx(graph)
#nx.draw(nx_graph)

# %%

