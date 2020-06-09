# Tutorial on Topological Data Analysis

This Jupyter-note book is prepared for the online event:
[TDA for Applications: Tutorial and Workshop](https://sites.google.com/view/tda-application-tutorial/)
being held on 18,19 June 2020.

Click [this link](https://colab.research.google.com/github/shizuo-kaji/TutorialTopologicalDataAnalysis/blob/master/TopologicalDataAnalysisWithPython.ipynb) to open the Jupyter notebook in Google Colaboratory.


## NLP Example (vectorise and visualise)
As an example of Natural Language Processing, we look at maths papers on arXiv.
This example runs only locally and not on Google Colab.

I did not tune parameters seriously, but I hope this erves as a starting point for other NLP applications.

- install gensim and nltk, if you do not have them
```
  conda install gensim nltk
  pip install kmapper
```
- install [Mehta](https://github.com/miku/metha): e.g., download [binary](https://github.com/miku/metha/releases) and
```
  sudo apt install ./metha_?.?.??_amd64.deb
```
- Download Metadata from arXiv (takes time)
```
  metha-sync -format arXiv -set math -base-dir .metha http://export.arxiv.org/oai2
```
- Learn a vector representation of papers using Doc2Vec and visualise using Mapper
```
  python metha2df.py -o arxiv_mapper.html
```
At the initial run, it produces picked dataframe "math_2007.pkl" containing Metadata.
Then, Doc2Vec model files "doc2vec_arxiv.model" and "doc2vec_arxiv.model.docvecs.vectors_docs.npy".
This process takes long.
For the second run, it loads learnt model and just produce the visualisation.

- Open arxiv_mapper.html in a browser


