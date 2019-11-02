# Naive-Bayes-Classification
CSC 412 Fall 2019 Programming Assignment 2
For Parts 1 and 2 of the assignment

Part 1 is uploaded as of 11/2/2019.
Part 2 is being worked on at the time of this writing.

This is a digit classifier coded in C++. I could have done it in MATLAB or Python, but because I wasn't sure how the probability formula was supposed to work, I used the language I was most familiar with. Note that in the implementation of the probability formula to find posterior probabilities for each class (digit), I had to add the logs of the probabilities instead of the multiplying the probabilities to avoid underflow; then since for a probability (which is a value between 0 and 1) a lower probability increases the negative of its log and a higher probability decreases the negative of its logs, I added the negative of the logs and then subtracted the total from 1 to find a proportional probability in regard to each class for each class.

Run in Visual Studio 2019. Keep training and testing data files in same directory that the program runs in (the same directory where the code files are located).

Training and Testing Data Files Descriptions

trainingimages: 5000 training digits, around 500 samples from each digit class. Each digit is of size 28x28, and the digits are concatenated together vertically. The file is in ASCII format, with three possible characters. ' ' means a white (background) pixel, '+' means a gray pixel, and '#' means a black (foreground) pixel. 

traininglabels: a vector of ground truth labels for every digit from trainingimages.

testimages: 1000 test digits (around 100 from each class), encoded in the same format as the training digits.

testlabels: ground truth labels for testimages.
