## NLP Example: Vectorization and Visualization of Math Papers on arXiv  

In this example, we analyze math papers on arXiv to demonstrate **Natural Language Processing (NLP)** techniques.  

> **Note:** This example runs only locally and is not compatible with Google Colaboratory.  

While the parameters have not been fine-tuned, this serves as a starting point for more serious NLP applications.  

### Steps to Run the Example  

1. **Install Required Libraries**  
   Ensure you have `gensim`, `nltk`, and `kmapper` installed:  
   ```bash
   conda install gensim nltk
   pip install kmapper
   ```

2. Install metha

For Linux users:
Download the binary release and install it:
  ```bash
  sudo apt install ./metha_?.?.??_amd64.deb
  ```

3. Download Metadata from arXiv

Use the following command to synchronize metadata from arXiv (this may take time):

   ```bash
    metha-sync -format arXiv -set math -base-dir .metha http://export.arxiv.org/oai2
   ```

4. Learn Vector Representations and Visualize

Generate a vector representation of papers using Doc2Vec and visualize it with Mapper:

  ```bash
  python metha2df.py -o arxiv_mapper.html
  ```

  The first run creates a pickled DataFrame (math_2007.pkl) containing metadata, along with Doc2Vec model files:

    -- doc2vec_arxiv.model
    -- doc2vec_arxiv.model.docvecs.vectors_docs.npy

  This process may take a long time.

  From the second time, the pre-trained model will be loaded, and only the visualization will be generated.

5. Open the Visualization

Open arxiv_mapper.html in your browser to explore the results.
