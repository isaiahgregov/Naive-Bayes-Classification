//This class is for Part 1 of the programming assignment: Single pixels as features

#pragma once
class naiveBayesDigitClassifier
{
private:
	float smoothingConstant;

	//Ten(10) 28 x 28 arrays that records the probability that the i * j - th pixel belongs to a digit,
	//where each digit corresponds to one of the ten arrays.
	float pixelProbability[10][28][28];

	//An array of ten(10) float values, where each value is the prior class probability.The prior class
	//is the digit.This is found by dividing the number of training examples per digit by 5000, for each digit.
	float priorProbability[10];

	//An array of 1000 x 10 float values, where each value corresponds to the posterior probability
	//for that digit (one of the 10 digits) for each test digit (each of the 1000 test digits)
	float testDigitProbability[1000][10];

	//A 1000 int array to hold the test labels(and therefore the correct answers for the test data)
	//Note : May have to initialize this array or any other arrays or static objects to 0 for each element,
	//depending on programming language.
	int testLabels[1000];

	//A float value that is used to record the model’s classification success rate of the test digits
	float totalClassificationRate;

	//An array of 10 float values that is used to record the model’s classification success rate for each digit
	float digitClassificationRate[10];

	//A ten(10) - element array that keeps count of the number of test examples for each digit,
	//which corresponds to each element of the array.
	int numOfTestExamples[10];

	//This is a 10 x 10 matrix whose entry in row r and column c is the percentage of test images from class r
	//that are classified as class c.
	float confusionMatrix[10][10];

	void evaluateModel();
	void printEvaluation();
	void printClassificationRateAndConfusionMatrix();
	void printFeatureLikelihoodsAndOddsRatios();
	void displayFeatureLikelihoodsAndOddsRatiosMaps(float pixelOdds[10][10][28][28], int maxRow, int maxCol);

public:
	naiveBayesDigitClassifier(float fSmoothingConstant);
	~naiveBayesDigitClassifier();

	void trainModel();
	void testModel();
};
