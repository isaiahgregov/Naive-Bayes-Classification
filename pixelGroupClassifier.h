//This class is for Part 2 of the programming assignment: Pixel groups as features
//
//Objective of this class:
//Find test set accuracies (total classification rate and digit classification rates)
//for disjoint patches of size 2*2, 2*4, 4*2, 4*4, and for overlapping patches of
//size 2*2, 2*4, 4*2, 4*4, 2*3, 3*2, 3*3.
//Find training and testing running time for different feature sets.

#pragma once
class pixelGroupClassifier
{
private:
	double smoothingConstant;
	
	//See below to see which numbers correspond to which feature sets.
	int featureSet;

	int n, m;
	
	bool featureSetDisjoint = false;
	bool featureSetOverlapping = false;

	//Ten(10) 28 x 28 arrays that records the probability that the i * j - th pixel belongs to a digit,
	//where each digit corresponds to one of the ten arrays.
	double pixelProbability[10][28][28];

	//An array of ten(10) double values, where each value is the prior class probability.The prior class
	//is the digit. This is found by dividing the number of training examples per digit by 5000, for each digit.
	double priorProbability[10];

	//An array of 1000 x 10 double values, where each value corresponds to the posterior probability
	//for that digit (one of the 10 digits) for each test digit (each of the 1000 test digits)
	double testDigitProbability[1000][10];

	//A 1000 int array to hold the test labels(and therefore the correct answers for the test data)
	//Note : May have to initialize this array or any other arrays or static objects to 0 for each element,
	//depending on programming language.
	int testLabels[1000];

	//A double value that is used to record the model’s classification success rate of the test digits
	double totalClassificationRate;

	//An array of 10 double values that is used to record the model’s classification success rate for each digit
	double digitClassificationRate[10];

	//A ten(10)-element array that keeps count of the number of test examples for each digit,
	//which corresponds to each element of the array.
	int numOfTestExamples[10];

	void evaluateModel();
	void printEvaluation();

public:
	/*
		Feature sets correspond to:

		1.  Disjoint pixel groups of size 2*2
		2.  Disjoint pixel groups of size 2*4
		3.  Disjoint pixel groups of size 4*2
		4.  Disjoint pixel groups of size 4*4
		5.  Overlapping pixel groups of size 2*2
		6.  Overlapping pixel groups of size 2*4
		7.  Overlapping pixel groups of size 4*2
		8.  Overlapping pixel groups of size 4*4
		9.  Overlapping pixel groups of size 2*3
		10. Overlapping pixel groups of size 3*2
		11. Overlapping pixel groups of size 3*3
	*/

	pixelGroupClassifier(double fSmoothingConstant, int featureSet);
	~pixelGroupClassifier();

	void trainModel();
	void testModel();
};

