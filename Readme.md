# Assignment 2 and Assignment 3: Natural Language Processing (CS613) - IIT Gandhinagar

## Course Overview
- The course provides an introduction to Natural Language Processing (NLP) in the B.Tech in Computer Science program at IIT Gandhinagar.
- Covers fundamental and advanced concepts including text processing, classification, clustering, summarization, generation, and deep learning for NLP.
- Hands-on programming assignments enhance practical understanding.

## Team Members
- Dewansh Singh Chandel
- Vraj Shah
- Suteekshna Misha
- Chandrabhan Patel
- Vedant Chichmalkar
- Kevin Shah
- Hetvi Patel
- Lipika Rajpal
- Sujal Patel

## Libraries Used
- Numpy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- NLTK
- Transformers

## Ngrams Assignment
- **Data Acquisition**: Obtain dataset from specified source.
- **Preprocessing**: Tokenize corpus into sentences using NLTK sentence tokenizer.
- **Data Splitting**: Randomly split corpus into 80% training and 20% validation sets.
- **Model Training and Validation**:
  - Train language models (LMs) including Unigram, Bigram, Trigram, and Quadgram on training set.
  - Validate trained models on validation set and report perplexity scores.
  - Implement LMs from scratch, without using existing NLP libraries.
- **Laplace Smoothing**:
  - Apply Laplace smoothing to trained LMs.
  - Compare perplexity scores before and after smoothing.
- **Justification**: Provide observations and justifications for impact of smoothing on perplexity scores.
- **Additional Smoothing Techniques**:
  - Choose and implement two other smoothing techniques (Additive, Good Turing, or Kneser-Ney).
  - Train same n-gram LMs using these techniques.
  - Discuss understanding and implications of using different smoothing techniques.

## Pretraining and Fine-tuning LLM Assignment
- **Parameter Calculation**: Calculate number of parameters of selected BERT model using provided code. Compare with parameters reported in corresponding paper.
- **Pretraining**: Pretrain selected BERT model on training split of 'wikitext-2-raw-v1' dataset for 5 epochs with chosen hyperparameters.
- **Perplexity Scores**: Compute and report perplexity scores using inbuilt function on test split of 'wikitext-2-raw-v1' dataset for each epoch of pretraining. Discuss trends in scores.
- **Model Deployment**: Push pre-trained BERT model to Hugging Face model repository.
- **Fine-tuning**: Fine-tune pretrained BERT model on Classification (SST-2) and Question-Answering (SQuAD) tasks using 80:20 train-test split with seed 1. Perform fine-tuning only on train split.
- **Metric Calculation**:
  - For Classification (SST-2): Calculate Accuracy, Precision, Recall, and F1 score.
  - For Question-Answering (SQuAD): Calculate squad_v2, F1 score, METEOR, BLEU, ROUGE, and exact-match.
- **Parameter Comparison**: Calculate number of parameters in model after fine-tuning and compare with pre-trained model. Determine if parameter count remains same after fine-tuning.
- **Model Deployment (Fine-tuned)**: Push fine-tuned BERT model to Hugging Face model repository.
- **Comments and Rationale**:
  - Provide comments and rationale behind observed model performance.
  - Analyze understanding gained from comparing parameters between pretraining and fine-tuning stages.
