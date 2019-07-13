# Overview

I am creating a series of tutorials where I implement things from scratch and try to make things as easy to understand as possible. Code wise I am trying to keep things very simple. Long term ... I have no idea where this is headed, but for now I am starting with one of my favorite areas which is distributional semantics. Many people think all of this started with word2vec. It didn't. There is a lot of history and various methods leading up to where we are now. To me at least, the current innovations are even more interesting when they are placed in their historical context.

For now I plan to cover the following:

Hyperspace Analogue to Language (HAL)<br/>
Latent Semantic Analysis (LSA)<br/>
Random Indexing<br/>
Binary Spatter Code<br/>
Topic Modeling (LDA)<br/>
word2vec<br/>
At some point I will ad basic tutorials on vector operations and fundamentals.

Note: This is very early work in progress.

# Implemented modules

## Random Indexing

Status: ~80% complete

Random Indexing is a method for creating embeddings. Random Indexing emerged as an alternative to LSA, which was very computationally demanding at the time. Random Indexing utilizes random projection to perform the dimensionality reduction step instead of SVD as with LSA. Random projection is a very inexpensive step, which makes Random Indexing highly scalable. With Random Indexing you select the dimensionality of the vector (500-1000). You then randomly set a small number of the elements to +1 or -1. At this point you have projected whatever you are modeling into a reduced dimensional space. The training procedure is simple addition based on co-occurrence. The method is also very flexible and can be applied in many different ways. Below are the variations that I have implemented.

Random Indexing - This is the initial method. You treat a block of text as context and give it a unique identifier (could be sentence, abstract, entire document, etc.). You initialize the document vectors using random projection. For each term in the document add the document vector. This produces term vectors only.

Random Indexing with sliding window - It may not be advantageous to always use a document as context. Here I generate a context window around each term and use this for training.

Document-based Reflective Random Indexing (DRRI) - The Relective in Reflective Random Indexing simply means that more iterations are used in training. With DRRI you begin by applying random projection to the documents. For each term in the document add the document vector to the term vector. Next, for each document, add all of the term vectors for the terms in the document to the document vector. Finally, you perform the first step again and add the document vectors to the terms vectors for each term in the document. This generates both term and document vectors.

Term-based Reflective Random Indexing (TRRI) - This is similar to DRRI, but the order is different. Here you begin with using random projection on the term vectors.

Metadata Reflective Random Indexing (MRRI) - This implementation highlights the flexibility of Random Indexing. Here I build associations between terms in an abstract and document labels.
