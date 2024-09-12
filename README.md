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
![Model Accuracy](https://github.com/youssefokeil/SentimentAnalysis/blob/main/Sentiment%20Analysis%20Figures/tf_accuracy.png)

Model Loss
![Model Loss](https://github.com/youssefokeil/SentimentAnalysis/blob/main/Sentiment%20Analysis%20Figures/tf_loss.png)
## DL with Pytorch
### Model Definition
```
SentimentLSTM(
  (embed): Embedding(16000, 300, padding_idx=0)
  (lstm): LSTM(300, 100, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
  (conv): Conv1d(280, 100, kernel_size=(5,), stride=(1,))
  (pool): AdaptiveAvgPool1d(output_size=100)
  (fc1): Linear(in_features=100, out_features=16, bias=True)
  (fc2): Linear(in_features=16, out_features=1, bias=True)
)

```
### Model Performance

Model Accuracy
![Model Accuracy](https://github.com/youssefokeil/SentimentAnalysis/blob/main/Sentiment%20Analysis%20Figures/torch_accuracy.jpeg)

Model Loss
![Model Loss](https://github.com/youssefokeil/SentimentAnalysis/blob/main/Sentiment%20Analysis%20Figures/torch_loss.jpeg)
## Classical Approach
### Linear SVM
`clf=svm.LinearSVC()`
### Model Performance
Model Accuracy
`The accuracy of our model is 53.062%`


Classification Report
```
Model Loss
              precision    recall  f1-score   support

    negative       0.53      0.61      0.57      4008
     positve       0.54      0.45      0.49      3992

    accuracy                           0.53      8000
   macro avg       0.53      0.53      0.53      8000
weighted avg       0.53      0.53      0.53      8000
```
