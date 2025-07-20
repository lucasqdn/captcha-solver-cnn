# captcha-solver

A CNN machine learning model that solves 4-letters captcha by individually separating the letters

The model works by:

1. Separating the captcha into individual letters
2. Each letter is then put through a CNN classification model in order to detect the letter
3. Each letters are then put together to form the final captcha.

This model yields a 99.65% accuracy while in training.

## Installation and train

!!! This project requires a python environment already installed

### Clone the project

Clone the project by first cloning the github repo into your local machine
Then, navigate to the project folder on your local machine using `cd`

### Training

First run requirements.txt to install the dependencies

```
pip install -r requirements.txt
```

After installing the necessary dependencies, run main.py in order to train the model

```
python main.py
```
