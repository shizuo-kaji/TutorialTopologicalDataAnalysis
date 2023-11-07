## NLP Example (vectorise and visualise)
As an example of Natural Language Processing, we look at maths papers on arXiv.
This example runs only locally and not on Google Colab.

I did not tune parameters seriously, but I hope this example serves as a starting point for more serious NLP applications.

- install gensim and nltk, if you do not have them
```
  conda install gensim nltk
  pip install kmapper
```
- install [metha](https://github.com/miku/metha): e.g., (if you are a linux user) download [binary](https://github.com/miku/metha/releases) and
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
