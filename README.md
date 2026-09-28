# Text Classification with Pre-trained Word Embeddings

Final project for **COMP 550 (Natural Language Processing)** at McGill University, Fall 2021, by Javin Liu, Yuxuan Tian and Zhixin Xiong.

We compare how different word embeddings affect an LSTM text classifier. We keep the model the same and swap only its embedding layer, then measure the results on three datasets that differ in size, number of labels and vocabulary.

📄 **Full report:** [Comp_550_Final_Project_Javin_Ethan_Zhixin.pdf](Comp_550_Final_Project_Javin_Ethan_Zhixin.pdf)

## Approach

Every model uses the same PyTorch architecture: **embedding → LSTM → linear layer → dropout**. Only the embedding changes:

| Variant | Embedding layer |
|---|---|
| **Vanilla** | Trainable `nn.Embedding`, randomly initialized (the baseline) |
| **Word2Vec** | Pre-trained [Google News Word2Vec](https://github.com/mmihaltz/word2vec-GoogleNews-vectors) vectors (300 dimensions) |
| **GloVe (50)** | Pre-trained [GloVe 6B](https://nlp.stanford.edu/projects/glove/) vectors (50 dimensions) |
| **GloVe (300)** | Pre-trained [GloVe 840B](https://www.kaggle.com/takuok/glove840b300dtxt) vectors (300 dimensions), for a fair comparison with Word2Vec |
| **BERT** | BERT word embeddings |

For the COVID tweets, we also removed `@` mentions and emoji, which would otherwise become unknown tokens.

## Datasets

- [**Spam Detection**](https://www.kaggle.com/kredy10/simple-lstm-for-text-classification/data): 5,169 SMS messages, labelled spam or not spam. A small, simple binary task.
- [**Women's E-Commerce Clothing Reviews**](https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews): product reviews with a 5-level rating to predict.
- [**Coronavirus Tweets**](https://www.kaggle.com/datatattle/covid-19-nlp-text-classification): sentiment of tweets about COVID-19. It's full of new words (*sanitizer*, *coronavirus*) that pre-trained embeddings from before 2020 have never seen.

## Results

Scores on each dataset (from the report). The best score in each table is in bold.

**Spam detection**

| Embedding | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Vanilla | 0.971 | 0.971 | 0.971 | 0.971 |
| Word2Vec | 0.978 | 0.979 | 0.978 | 0.978 |
| BERT | 0.978 | 0.974 | 0.928 | 0.949 |
| GloVe (50) | 0.950 | 0.949 | 0.950 | 0.949 |
| GloVe (300) | **0.979** | **0.979** | **0.979** | **0.979** |

**Clothing reviews**

| Embedding | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Vanilla | **0.720** | **0.701** | **0.716** | **0.705** |
| Word2Vec | 0.647 | 0.615 | 0.647 | 0.628 |
| BERT | 0.634 | 0.606 | 0.634 | 0.616 |
| GloVe (50) | 0.663 | 0.625 | 0.663 | 0.634 |
| GloVe (300) | 0.659 | 0.618 | 0.659 | 0.623 |

**COVID tweets**

| Embedding | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Vanilla | 0.640 | 0.640 | 0.641 | 0.639 |
| Word2Vec | **0.780** | **0.781** | **0.780** | **0.780** |
| BERT | 0.766 | 0.765 | 0.765 | 0.766 |
| GloVe (50) | 0.661 | 0.622 | 0.661 | 0.628 |
| GloVe (300) | 0.710 | 0.710 | 0.710 | 0.710 |

### What the numbers show
- **No single embedding wins everywhere.** A different embedding was best on each dataset: GloVe (300) on spam, the vanilla embedding on clothing reviews, and Word2Vec on COVID tweets.
- **Pre-trained embeddings helped most on the COVID tweets.** Word2Vec and BERT beat the vanilla embedding by 13–14 F1 points there.
- **On clothing reviews, the vanilla embedding did best.** Every pre-trained embedding scored lower than the trainable one.
- **Spam detection is easy for every variant** (F1 0.95–0.98). BERT's recall there was noticeably lower than the rest (0.928, against roughly 0.95–0.98).
- **Bigger GloVe vectors helped.** Moving from 50 to 300 dimensions improved GloVe on spam and COVID tweets, and it scored about the same on clothing reviews.

The discussion in the PDF interprets the results more favourably to BERT; the tables above are the measured numbers.

## Repository contents

| Notebook | Embedding | Dataset |
|---|---|---|
| `Vanila_LSTM_spam.ipynb` | Vanilla | Spam |
| `Normal_LSTM_Clothing_ipynb”.ipynb` | Vanilla | Clothing reviews |
| `Vanila_LSTM_Covid.ipynb`, `Vanilla_LSTM_Covid.ipynb` | Vanilla | COVID tweets |
| `Word2Vec_LSTM_spam.ipynb` | Word2Vec | Spam |
| `Word2Vec_BERT_LSTM_Clothing.ipynb` | Word2Vec and BERT | Clothing reviews |
| `Glove_LSTM_spam.ipynb` | GloVe | Spam |
| `Glove_LSTM_Clothing.ipynb` | GloVe | Clothing reviews |
| `Glove_LSTM_Covid.ipynb` | GloVe | COVID tweets |

## Running the notebooks

The notebooks were written for Google Colab or Jupyter, with PyTorch.

1. Download the dataset for the notebook you want to run (links above).
2. For the pre-trained variants, download the embedding files:
   - [glove.6B.50d.txt](https://www.kaggle.com/watts2/glove6b50dtxt)
   - [glove.840B.300d.txt](https://www.kaggle.com/takuok/glove840b300dtxt)
   - [GoogleNews Word2Vec vectors](https://github.com/mmihaltz/word2vec-GoogleNews-vectors)

   BERT weights download automatically through Hugging Face `transformers`.
3. Update the file paths at the top of the notebook, and run all cells.

## Authors

- **Javin Liu**: BERT embedding experiments (spam and clothing reviews), the LaTeX report
- **Yuxuan Tian**: vanilla LSTM on clothing reviews, Word2Vec LSTM
- **Zhixin Xiong**: GloVe LSTM on COVID tweets and spam
