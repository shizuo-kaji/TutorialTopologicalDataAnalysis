# Tutorial on Topological Data Analysis
Written by Shizuo KAJI

This Jupyter-note book is prepared for the online event:
[TDA for Applications: Tutorial and Workshop](https://sites.google.com/view/tda-application-tutorial/)
being held on 18,19 June 2020.

## Main Examples
Click [this link](https://colab.research.google.com/github/shizuo-kaji/TutorialTopologicalDataAnalysis/blob/master/TopologicalDataAnalysisWithPython.ipynb) to open the Jupyter notebook in Google Colaboratory.

This includes
- Feature extraction using persistent homology from various types of data (point cloud, graph, image, volume, time-series)
- Regression/Classification using topological features
- Dimension reduction preserving topological features
- Visualisation revealing the shape of data

## Deep Learning X TDA
How Deep Learning and Persistent homology can be combined is demonstrated [here](https://github.com/shizuo-kaji/HomologyCNN).

## NLP Example (vectorise and visualise)
As an example of Natural Language Processing, we look at maths papers on arXiv.
This example runs only locally and not on Google Colab.

I did not tune parameters seriously, but I hope this example serves as a starting point for more serious NLP applications.

- install gensim and nltk, if you do not have them
```
  conda install gensim nltk
  pip install kmapper
```
- install [mehta](https://github.com/miku/metha): e.g., (if you are a linux user) download [binary](https://github.com/miku/metha/releases) and
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
At the initial run, it produces a pickled dataframe "math_2007.pkl" containing Metadata,
and Doc2Vec model files "doc2vec_arxiv.model" and "doc2vec_arxiv.model.docvecs.vectors_docs.npy".
This process takes long.
For the second run, it loads the learnt model and just produces the visualisation.
For a good visualisation, parameters for both Doc2vec and Mapper should be tuned.

- Open arxiv_mapper.html in a browser

## Reference

- [Persistent homology - An introduction via interactive examples](https://iuricichf.github.io/ICT/index.html) provides a quick access to the theory of persistent homology.

