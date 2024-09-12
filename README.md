# TL;DR
I use a subset of the IMDB dataset to do sentiment analysis. I used a naive approach to encode words, but ended up with nearly 140,000 words, even after removing stop words, punctuation, html tags and lemmatizing the text.
So I use the Word2Vec to embed my vocbulary, wnded up with a 16,000 word dictionary. I use then BiLSTM, Conv1D and Dense Layers to predict.
## DL with TensorFlow
### Model Definition
```
Model: "sequential_6"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_3 (Embedding)      (None, 280, 300)          4800000   
_________________________________________________________________
bidirectional_12 (Bidirectio (None, 280, 200)          320800    
_________________________________________________________________
bidirectional_13 (Bidirectio (None, 280, 200)          240800    
_________________________________________________________________
conv1d_6 (Conv1D)            (None, 276, 100)          100100    
_________________________________________________________________
global_average_pooling1d_1 ( (None, 100)               0         
_________________________________________________________________
dense_12 (Dense)             (None, 16)                1616      
_________________________________________________________________
dense_13 (Dense)             (None, 1)                 17        
=================================================================
Total params: 5,463,333
Trainable params: 663,333
Non-trainable params: 4,800,000
_________________________________________________________________
```
### Model Performance
Model Accuracy
![Model Accuracy]("Sentiment Analysis Figures/tf_accuracy.png")
Model Loss
![Model Loss]("Sentiment Analysis Figures/tf_loss.png")
