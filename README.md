# Sentiment-Analysis

# Git Preliminary

1. git clone https://github.com/zhengwang125/Sentiment-Analysis.git
2. git add "your modified"
3. git status 
4. git commit -m "your commitments"
5. git push origin master
6. git pull origin master

# Downloading Data

Before running the python code, you'll first need to download all data we'll be using. This data is located in the Combined_News_DJIA1.csv. We will process these in the same directory as source code. As always, the first step is to clone the repository.

# Requirements and Installation

In order to run the Python code, you'll need the following libraries.

[TensorFlow](http://www.tensorflow.org/) version 1.1 (Currently not supported for 1.2, 1.3, 1.4 etc but if someone wants to submit a pull request, I'd be open to that.)
[NumPy](http://www.numpy.org/)
[Matplotlib](http://www.matplotlib.org/)
Installing [Anaconda Python](https://www.anaconda.com/) and [Keras](https://keras.io/)
[Sklearn](http://scikit-learn.org/stable/)
The easiest way to install keras (Backend is TensorFlow) as well as NumPy, Matplotlib, Sklearn and so on is to start with the Anaconda Python distribution.

Follow the installation instructions for Anaconda Python. We recommend using Python 3.6.

Follow the platform-specific TensorFlow installation instructions. Be sure to follow the "Installing with Anaconda" process, and create a Conda environment named tensorflow.

If you aren't still inside your Conda TensorFlow environment, enter it by opening your terminal and typing

source activate tensorflow
If you haven't done so already, download and unzip this entire repository from GitHub, either interactively, or by entering

git clone https://github.com/zhengwang125/Sentiment-Analysis.git
Use cd to navigate into the top directory of the repo on your machine

# Deep Learning

**Sentiment Analysis** is a classic problem in RNN with LSTM model. We try to implement it in keras framework and we also prepared a tensorflow version. The best accuracy is about 47.9%.

# Statistical Machine Learning

**Random Forest**, **K Nearest Neighbors**, **SVM**, **Voting** , **Logistic Regression** are also used for this prediction. More details you can found in homepage of [Scikit-learn Classification](http://scikit-learn.org/stable/supervised_learning.html#supervised-learning)

Algorithms | Accuracy
--------   | ---
Random Forest | 50.53%
KNN    | 55.82%
SVM     | 48.14%
Voting | 50.26%
Logistic Regression | 48.94%
