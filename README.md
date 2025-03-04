# Natural Language Processing with Disaster Tweets

This repository contains my solution for the [Kaggle "Natural Language Processing with Disaster Tweets" competition](https://www.kaggle.com/c/nlp-getting-started).

## Project Overview

The goal of this competition is to predict which tweets are about real disasters and which ones are not. This is a binary classification problem in the domain of Natural Language Processing (NLP).

## Dataset

The dataset consists of:
- 7,613 tweets (training set)
- 3,263 tweets (test set)
- Each tweet is labeled as either a disaster tweet (1) or not (0)
- Additional features include the tweet's location and keyword

## Repository Structure

- `disaster_tweets_EDA.ipynb`: Exploratory Data Analysis of the dataset
- `disaster_tweets_modeling.ipynb`: Model building, training, and evaluation
- `disaster_tweets_submission.ipynb`: Final model and Kaggle submission generation
- `requirements.txt`: List of packages required to run the notebooks
- `sample_predictions.csv`: Sample of test data predictions for inspection
- `submission.csv`: Final predictions for Kaggle submission

## Approach

### 1. Data Preprocessing

- Text cleaning (removal of URLs, special characters, stopwords)
- Stemming/lemmatization
- Tokenization and sequence padding
- Feature engineering (text length, word count, etc.)

### 2. Model Architecture

I experimented with several deep learning architectures:
- Simple LSTM models
- Bidirectional LSTM models
- Stacked LSTM architectures with global max pooling
- Combined models with additional text features

### 3. Hyperparameter Tuning

- Optimized embedding dimensions, LSTM units, and dropout rates
- Tested different learning rates and batch sizes
- Used early stopping and learning rate reduction to prevent overfitting

### 4. Results

- Best validation accuracy: X.XX
- Best F1 score: X.XX
- Kaggle competition score: X.XXX

## How to Run

1. Clone this repository:
```
git clone https://github.com/yourusername/disaster-tweets-nlp.git
cd disaster-tweets-nlp
```

2. Install the required packages:
```
pip install -r requirements.txt
```

3. Download the dataset from Kaggle:
   - Go to the [competition page](https://www.kaggle.com/c/nlp-getting-started)
   - Download the data files
   - Place them in the project directory

4. Run the notebooks in the following order:
   - `disaster_tweets_EDA.ipynb`
   - `disaster_tweets_modeling.ipynb`
   - `disaster_tweets_submission.ipynb`

## Key Insights

- Bidirectional LSTMs consistently outperformed simple LSTMs for this task
- Adding statistical text features improved model performance
- The most important features for classification included keywords related to emergencies, natural disasters, and crisis vocabulary
- Text preprocessing, especially removing noise like URLs and special characters, significantly improved model performance


# Conclusion

In this project, we tackled the Kaggle "Natural Language Processing with Disaster Tweets" competition, which involved classifying tweets as either describing a real disaster (1) or not (0). Let's summarize our approach, findings, and potential future improvements.

## Key Findings

1. **Text Preprocessing Impact**:
   - Removing stopwords and special characters significantly improved model performance
   - Stemming helped reduce vocabulary size without sacrificing too much semantic meaning

2. **Model Architecture Insights**:
   - Bidirectional LSTMs consistently outperformed unidirectional LSTMs
   - Stacked architectures showed better feature extraction capabilities
   - Adding text-based features provided additional signals that improved classification accuracy

3. **Hyperparameter Effects**:
   - Larger embedding dimensions generally led to better performance but with diminishing returns
   - Dropout rates between 0.3-0.5 helped prevent overfitting
   - Learning rates around 0.0005 provided the best convergence behavior

## What Worked Well

- **Bidirectional LSTM**: Captured context from both directions, improving understanding of tweet semantics
- **Word embeddings**: Effectively transformed text into numerical representations while preserving semantic relationships
- **Additional text features**: Complemented the semantic embeddings with statistical information about the tweets
- **Early stopping**: Prevented overfitting and helped identify the optimal training duration

## What Didn't Work as Expected

- **Simple LSTM models**: Struggled to capture complex relationships in the text
- **Very deep architectures**: Showed signs of overfitting despite dropout layers
- **Very high learning rates**: Led to unstable training and suboptimal convergence

## Future Improvements

1. **Advanced Preprocessing**:
   - Experiment with lemmatization instead of stemming
   - Implement more sophisticated text cleaning techniques
   - Use named entity recognition to identify disaster-related entities

2. **Enhanced Model Architectures**:
   - Implement transformer-based models like BERT, RoBERTa, or DistilBERT
   - Explore attention mechanisms to focus on the most relevant parts of tweets
   - Create ensemble models combining different architectures

3. **Feature Engineering**:
   - Use pretrained word embeddings (GloVe, Word2Vec, FastText)
   - Extract sentiment features using pretrained sentiment analyzers
   - Incorporate keyword importance through TF-IDF weighting

4. **Training Strategies**:
   - Implement k-fold cross-validation for more robust model evaluation
   - Use data augmentation techniques specific to text data
   - Experiment with more sophisticated optimizers and learning rate schedules

## Final Thoughts

This project demonstrated the effectiveness of deep learning approaches for natural language processing tasks, particularly text classification. LSTM-based architectures proved capable of capturing the semantic content needed to distinguish between disaster and non-disaster tweets. The combined model with additional features showed that incorporating domain knowledge and statistical information about texts can enhance pure deep learning approaches.

The techniques used here are transferable to many other NLP classification tasks, and the insights gained about text preprocessing, model architecture design, and hyperparameter tuning will be valuable for future text analysis projects.


## References

1. Kaggle Competition: [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started)
2. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.
3. TensorFlow documentation: [https://www.tensorflow.org/api_docs/python](https://www.tensorflow.org/api_docs/python)
