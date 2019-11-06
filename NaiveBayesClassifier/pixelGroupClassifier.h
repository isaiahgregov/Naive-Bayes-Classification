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
	float smoothingConstant;

	//See below to see which numbers correspond to which feature sets.
	int featureSet;

	int m, n;

	bool featureSetDisjoint = false;
	bool featureSetOverlapping = false;

	//Ten (10) … x … x 2^(m * n)) arrays that records the probability that the i*j-th pixel group belongs to a digit,
	//where each digit corresponds to one of the ten arrays.
	//If disjoint:
	//	double pixelGroupProbability[10][28 / m][28 / n)];
	//Else if overlapping:
	//	double pixelGroupProbability[10][29 – m][29 – n)];
	float**** pixelGroupProbability = new float*** [10];

	//An array of ten (10) double values, where each value is the prior class probability. The prior class
	//is the digit. This is found by dividing the number of training examples per digit by 5000, for each digit.
	//Note: From training digits
	float priorProbability[10];

	//An array of 1000 x 10 double values, where each value corresponds to the posterior probability
	//for that digit (one of the 10 digits) for each test digit (each of the 1000 test digits)
	//Note: From test digits
	float posteriorProbability[1000][10];

	//A 1000 int array to hold the test labels(and therefore the correct answers for the test data)
	int testLabels[1000];

	//A double value that is used to record the model’s classification success rate of the test digits
	float totalClassificationRate;

	//An array of 10 double values that is used to record the model’s classification success rate for each digit
	float digitClassificationRate[10];

	//A ten(10)-element array that keeps count of the number of test examples for each digit,
	//which corresponds to each element of the array.
	int numOfTestExamples[10];

	int numOfFeatureValues;

	int numOfPixelGroupRow;
	int numOfPixelGroupCol;

	void evaluateModel();
	void printEvaluation();
	int getPixelGroupNumber(char nextTrainingImage[28][28], int topLeftRow, int topLeftCol);

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

	pixelGroupClassifier(float fSmoothingConstant, int featureSet);
	~pixelGroupClassifier();

	void trainModel();
	void testModel();
};

