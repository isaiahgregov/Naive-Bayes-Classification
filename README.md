CSC 412 Fall 2019 Programming Assignment 2
For Parts 1 and 2 of the assignment

This is a digit classifier coded in C++. I could have done it in MATLAB or Python, but because I wasn't sure how the probability formula was supposed to work, I used the language I was most familiar with to start. Note that in the implementation of the probability formula to find posterior probabilities for each class (digit), I had to add the logs of the probabilities instead of the multiplying the probabilities to avoid underflow; then since for a probability (which is a value between 0 and 1) a lower probability increases the negative of its log and a higher probability decreases the negative of its logs, I added the negative of the logs and then subtracted the total from 1 to find a proportional probability in regard to each class for each class.

Run in Visual Studio 2019. Keep training and testing data files in same directory that the program runs in (the same directory where the code files are located). Run as an x64 program (program for x64 computer) or you will get a bad allocation exception when running pixel group feature number 8 (Overlapping 4x4), since this pixel group feature set requires the most memory; so in Visual Studio, select the x64 option for debugging to run it.

Note: The posterior probabilities of classes do not sum to 1 for each test digit. This is at least because logs were used to avoid underflow, and so the scale had to be converted into a number from 0 to 1. I did this by doing this: the max log was found, all posterior probabilities, initally being found as the sum of the logs of the likelihoods, were divided by the max log. So the posterior "probabilities" are proportionally accurate, but their scale is not from 0 to 1. I'd have to think of a way to scale them correctly, perhaps by taking the smallest and largest posterior probabilities and... scaling all probabilities to be from 0 to 1, given more time. Log probability is a subject in Computer Science that seeks to solve these problems:

https://en.wikipedia.org/wiki/Log_probability

Training and Testing Data Files Descriptions

trainingimages: 5000 training digits, around 500 samples from each digit class. Each digit is of size 28x28, and the digits are concatenated together vertically. The file is in ASCII format, with three possible characters. ' ' means a white (background) pixel, '+' means a gray pixel, and '#' means a black (foreground) pixel.

traininglabels: a vector of ground truth labels for every digit from trainingimages.

testimages: 1000 test digits (around 100 from each class), encoded in the same format as the training digits.

testlabels: ground truth labels for testimages.
