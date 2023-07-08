# deep-learning-challenge
## Alphabet Soup Charity - Neural Network Analysis Report
## Overview of the Analysis
The purpose of this analysis is to use deep learning to interpret the dataset provided by Alphabet Soup's Charity. Alphabet Soup's business team has compiled a list of over 34,000 organizations that have received funding from Alphabet Soup over the years. This report aims to create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup, using the features in the dataset.

## Results
## Data Preprocessing
The target variable for this model is IS_SUCCESSFUL. It indicates whether the money used by the organization was used effectively.
The feature variables for this model include 'APPLICATION_TYPE', 'AFFILIATION', 'CLASSIFICATION', 'USE_CASE', 'ORGANIZATION', 'STATUS', 'INCOME_AMT', 'SPECIAL_CONSIDERATIONS', and 'ASK_AMT'.
'EIN' (Employer Identification Number) and 'NAME' are neither targets nor features and were therefore dropped from the input data. They are unique identifiers and would not contribute to the model's ability to generalize and make predictions.
## Compiling, Training, and Evaluating the Model
The neural network model in this analysis consists of two hidden layers and an output layer. The first hidden layer contains 80 neurons, the second hidden layer has 30 neurons, and the output layer has 1 neuron. The activation function 'ReLU' is used in the hidden layers, and the 'sigmoid' function is used in the output layer. The 'ReLU' activation function is used to introduce non-linearity to the model, and it helps the model learn complex patterns in the data. The 'sigmoid' function is used in the output layer because it's a binary classification problem.
Unfortunately, the model did not reach its target performance. The accuracy of the model is approximately 66%, which is below the desired threshold.
Attempts to increase model performance could include:
Increasing the number of neurons or adding more hidden layers.
Applying different activation functions, such as tanh or leaky ReLU.
Increasing the number of epochs (iterations) in the training phase.
Using different optimization algorithms.
Using regularization techniques to prevent overfitting.
## Summary
In conclusion, the deep learning model had limited success predicting whether applicants will be successful if funded by Alphabet Soup, achieving an accuracy of approximately 66%. Considering that the model performance was below the desired threshold, alternative models could be considered.

As a recommendation, other machine learning models like Random Forest or Gradient Boosting could be considered for this classification problem. These ensemble methods work well for binary classification problems, are not as susceptible to outliers, and are less prone to overfitting than deep learning models. Additionally, they can provide feature importance, which can be very beneficial when interpreting the results.

Another recommendation would be to try different architectures of neural networks, maybe using more complex ones, if computational power allows it. We could try adding more layers, using different types of layers (like convolutional or recurrent layers, if they make sense in the context), or changing the number of neurons in the layers.

Lastly, if Alphabet Soup has more specific information about what makes an organization use the funds effectively, feature engineering could help improve the performance of any model used.
