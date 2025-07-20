# captcha-solver

A CNN machine learning model that solves 4-letters captcha by individually separating the letters

The model works by:

1. Separating the captcha into individual letters
2. Each letter is then put through a CNN classification model in order to detect the letter
3. Each letters are then put together to form the final captcha.

This model yields a 99.65% accuracy while in training.
