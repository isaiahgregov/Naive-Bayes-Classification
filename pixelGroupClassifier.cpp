#pragma once
#include "pixelGroupClassifier.h"
#include <iostream>
#include <chrono> 
#include <iostream>
#include <fstream>
#include <string> //to capture user input for error-checking
#include <sstream> //stringstream
#include <iomanip> //set precision
#include <math.h> //log

using namespace std::chrono;
using namespace std;

//*******************************************************************************
//--------------Constructors and Destructor-------------------------------------*
//*******************************************************************************

pixelGroupClassifier::pixelGroupClassifier(double fSmoothingConstant, int fFeatureSet)
{
	smoothingConstant = fSmoothingConstant;

	featureSet = fFeatureSet;

	featureSetDisjoint = false;
	featureSetOverlapping = false;

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

	switch (featureSet)
	{
	case 1:
		n = 2;
		m = 2;
		featureSetDisjoint = true;

		break;
	case 2:
		n = 2;
		m = 4;
		featureSetDisjoint = true;

		break;
	case 3:
		n = 4;
		m = 2;
		featureSetDisjoint = true;

		break;
	case 4:
		n = 4;
		m = 4;
		featureSetDisjoint = true;

		break;
	case 5:
		n = 2;
		m = 2;
		featureSetOverlapping = true;

		break;
	case 6:
		n = 2;
		m = 4;
		featureSetOverlapping = true;

		break;
	case 7:
		n = 4;
		m = 2;
		featureSetOverlapping = true;

		break;
	case 8:
		n = 4;
		m = 4;
		featureSetOverlapping = true;

		break;
	case 9:
		n = 2;
		m = 3;
		featureSetOverlapping = true;

		break;
	case 10:
		n = 3;
		m = 2;
		featureSetOverlapping = true;

		break;
	case 11:
		n = 3;
		m = 3;
		featureSetOverlapping = true;

		break;
	}

	totalClassificationRate = 0;

	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 28; j++)
		{
			for (int k = 0; k < 28; k++)
			{
				pixelProbability[i][j][k] = 0;
			}
		}
	}

	for (int i = 0; i < 10; i++)
	{
		priorProbability[i] = 0;
	}

	for (int i = 0; i < 1000; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			testDigitProbability[i][j] = 0;
		}
	}

	for (int i = 0; i < 1000; i++)
	{
		testLabels[i] = 0;
	}

	for (int i = 0; i < 10; i++)
	{
		digitClassificationRate[i] = 0;
	}

	for (int i = 0; i < 10; i++)
	{
		numOfTestExamples[i] = 0;
	}
}

pixelGroupClassifier::~pixelGroupClassifier()
{
}

//*******************************************************************************
//--------------Public Functions------------------------------------------------*
//*******************************************************************************

void pixelGroupClassifier::trainModel()
{
	//A temporary array of twenty - eight(28) 28 - char - length strings to update the pixelCountMatrices
	char nextTrainingImage[28][28];

	int nextLabel;

	//An array of ten(10) 28 x 28 float-type matrices to keep count of the number of times a pixel is 
	//“in the foreground” / “counts” for a digit, for each digit. Each element of this array corresponds
	//to each digit from 0 - 9. It is a float or double type for laplace smoothing.
	double pixelCountMatrices[10][28][28];

	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 28; j++)
		{
			for (int k = 0; k < 28; k++)
			{
				pixelCountMatrices[i][j][k] = 0;
			}
		}
	}

	//A ten(10) - element array that keeps count of the number of training examples for each digit, which
	//corresponds to each element of the array.
	int numOfTrainingExamples[10];

	for (int i = 0; i < 10; i++)
	{
		numOfTrainingExamples[i] = 0;
	}

	/*
	Algorithm:
		Loop 5000 times:
			Read in next 28 lines from digit data training images file into nextTrainingImage.
			Read in the next digit from the training labels file into nextLabel.
			numOfTrainingExamples[nextLabel] += 1;
			For each i*j-th element that contains the char character “+” or “#” in nextTrainingImage,
			increment the element for pixelCountMatrices[nextLabel] by one (1).
		Loop 28 x 28 x 10 = 7840 times(Or do a double loop, the first 10 times, the second 28 x 28 times…):
			Update pixelProbability for each pixel counted in pixelCountMatrices for each digitand laplace smooth,
			implementing the conditional probability formula, where P(Fij | class) is updated in pixelProbability:
				P(pixelij | class) = (# of times pixel(i, j) is counted in training examples from this
				class = pixelCountMatrices[digit being classified(determined by the loop number)])
				+ smoothingConstant / (Total # of training examples from this class =
				numOfTrainingExamples[digit being classified(determined by the loop number)] + 2 * smoothingConstant)
		Loop 10 times:
			For each loop, update each digit in priorProbability : priorProbability[loop number]
			= numOfTrainingExamples[loop number] / 5000
	*/

	//Open data files.
	ifstream dataFileInTrainingImages;
	dataFileInTrainingImages.open("trainingimages");
	ifstream dataFileInTrainingLabels;
	dataFileInTrainingLabels.open("traininglabels");

	//Strings used for processing input from data files.
	string nextLine1, nextLine2;

	for (int i = 0; i < 5000; i++)
	{
		for (int j = 0; j < 28; j++)
		{
			getline(dataFileInTrainingImages, nextLine1);

			for (int k = 0; k < 28; k++)
			{
				nextTrainingImage[j][k] = nextLine1[k];
			}
		}

		getline(dataFileInTrainingLabels, nextLine2);

		//Convert string to number.
		stringstream myStringstream(nextLine2);
		myStringstream >> nextLabel;

		numOfTrainingExamples[nextLabel] += 1;

		//For each i* j-th element that contains the char character “+” or “#” in nextTrainingImage,
		//increment the element for pixelCountMatrices[nextLabel] by one(1).
		for (int j = 0; j < 28; j++)
		{
			for (int k = 0; k < 28; k++)
			{
				if (nextTrainingImage[j][k] == '+' || nextTrainingImage[j][k] == '#')
				{
					pixelCountMatrices[nextLabel][j][k] += 1;
				}
			}
		}

		/*
		Loop 28 x 28 x 10 = 7840 times(Or do a double loop, the first 10 times, the second 28 x 28 times…):
			Update pixelProbability for each pixel counted in pixelCountMatrices for each digit and laplace smooth,
			implementing the conditional probability formula, where P(Fij | class) is updated in pixelProbability:

				P(pixelij | class) = (# of times pixel(i, j) is counted in training examples from this
				class = pixelCountMatrices[digit being classified(determined by the loop number)])
				+ smoothingConstant / (Total # of training examples from this class =
				numOfTrainingExamples[digit being classified(determined by the loop number)] + 2 * smoothingConstant)
		*/

		for (int j = 0; j < 10; j++)
		{
			for (int k = 0; k < 28; k++)
			{
				for (int l = 0; l < 28; l++)
				{
					pixelProbability[j][k][l] = (pixelCountMatrices[j][k][l] + smoothingConstant)
						/ (numOfTrainingExamples[j] + 2 * smoothingConstant);
				}
			}
		}

		/*
		Loop 10 times:
		For each loop, update each digit in priorProbability : priorProbability[loop number]
			= numOfTrainingExamples[loop number] / 5000
		*/

		for (int j = 0; j < 10; j++)
		{
			priorProbability[j] = (double)numOfTrainingExamples[j] / (double)5000;
		}
	}

	//Close data files.
	dataFileInTrainingImages.close();
	dataFileInTrainingLabels.close();
}

void pixelGroupClassifier::testModel()
{
	//A temporary array of twenty - eight(28) 28 - char - length strings to find that image’s posterior probability
	char nextTestImage[28][28];

	int nextLabel;

	/*
		Algorithm:
			Loop 1000 times (testDigitCount):
				Read in next 28 lines from digit data test images file into nextTestImage.
				Read in the next digit from the test labels file into testLabels[testDigitcount].
				numOfTestExamples[testLabels[testDigitCount]] += 1
				Implement the formula P(class) ∙ P(f1,1 | class) ∙ P(f1,2 | class) ∙ ... ∙ P(f28,28 | class) to find posterior probabilities:
				Comment: Change these values to logs as the assignment description says, if underflow occurs
				(and add instead—see assignment description for formula.).
					double pixelProduct = 1;
						Comment: pixelProduct is a temporary value to hold the latter portion of the product of the above formula.
					Loop 10 times (i):
						Loop 28 times (j):
							Loop 28 times (k):
								If nextTestImage[j, k] == “ “:
									pixelProduct = pixelProduct * (1 - pixelProbability[i, j, k])
								Else if nextTestImage[j, k] == “+“ or “#”:
									pixelProduct = pixelProduct * pixelProbability[i, j, k]
						testDigitProbability[testDigitCount, i] = priorProbability[i] * pixelProduct
			evaluateModel()
			printEvaluation()
	*/

	//Open data files.
	ifstream dataFileInTestImages;
	dataFileInTestImages.open("testimages");
	ifstream dataFileInTestLabels;
	dataFileInTestLabels.open("testlabels");

	//Strings used for processing input from data files.
	string nextLine1, nextLine2;

	double maxLog = 0;

	for (int i = 0; i < 1000; i++)
	{
		for (int j = 0; j < 28; j++)
		{
			getline(dataFileInTestImages, nextLine1);

			for (int k = 0; k < 28; k++)
			{
				nextTestImage[j][k] = nextLine1[k];
			}
		}

		getline(dataFileInTestLabels, nextLine2);

		//Convert string to number.
		stringstream myStringstream(nextLine2);
		myStringstream >> nextLabel;

		testLabels[i] = nextLabel;

		numOfTestExamples[testLabels[i]] += 1;

		/*
		Implement the formula P(class) ∙ P(f1,1 | class) ∙ P(f1,2 | class) ∙ ... ∙ P(f28,28 | class) to find posterior probabilities:
		Comment: Change these values to logs as the assignment description says, if underflow occurs
		(and add instead—see assignment description for formula.).
			double pixelProduct = 1;
				Comment: pixelProduct is a temporary value to hold the latter portion of the product of the above formula.
			Loop 10 times (i):
				Loop 28 times (j):
					Loop 28 times (k):
						If nextTestImage[j, k] == “ “:
							pixelProduct = pixelProduct * (1 - pixelProbability[i, j, k])
						Else if nextTestImage[j, k] == “+“ or “#”:
							pixelProduct = pixelProduct * pixelProbability[i, j, k]
				testDigitProbability[testDigitCount, i] = priorProbability[i] * pixelProduct

		Note that in the implementation of the probability formula to find posterior probabilities for each class (digit) below,
		I had to add the logs of the probabilities instead of the multiplying the probabilities to avoid underflow;
		then since for a probability (which is a value between 0 and 1) a lower probability increases the negative of
		its log and a higher probability decreases the negative of its logs, I added the negative of the logs and then
		subtracted the total from 1 to find a proportional probability in regard to each class for each class.
		*/

		for (int j = 0; j < 10; j++)
		{
			//pixelProduct is a temporary value to hold the latter portion of the product of the above formula.
			double pixelProduct = 0;

			for (int k = 0; k < 28; k++)
			{
				for (int l = 0; l < 28; l++)
				{
					if (nextTestImage[k][l] == ' ')
					{
						pixelProduct = pixelProduct - log(1 - pixelProbability[j][k][l]);
					}
					else if (nextTestImage[k][l] == '+' || nextTestImage[k][l] == '#')
					{
						pixelProduct = pixelProduct - log(pixelProbability[j][k][l]);
					}
				}
			}

			testDigitProbability[i][j] = pixelProduct + log(priorProbability[j]);

			if (testDigitProbability[i][j] > maxLog)
			{
				maxLog = testDigitProbability[i][j];
			}

			testDigitProbability[i][j] = testDigitProbability[i][j];
		}
	}

	for (int i = 0; i < 1000; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			testDigitProbability[i][j] = testDigitProbability[i][j] / maxLog;
			testDigitProbability[i][j] = 1 - testDigitProbability[i][j];
		}
	}


	//Close data files.
	dataFileInTestImages.close();
	dataFileInTestLabels.close();

	evaluateModel();
	printEvaluation();
}

//*******************************************************************************
//--------------Private Functions-----------------------------------------------*
//*******************************************************************************

void pixelGroupClassifier::evaluateModel()
{
	int numTotalTestDigitsCorrect = 0;
	int numEachTestDigitCorrect[10];

	for (int i = 0; i < 10; i++)
	{
		numEachTestDigitCorrect[i] = 0;
	}

	for (int i = 0; i < 1000; i++)
	{
		int locOfMax = 0;
		double max = 0;

		for (int j = 0; j < 10; j++)
		{
			if (testDigitProbability[i][j] > max)
			{
				max = testDigitProbability[i][j];
				locOfMax = j;
			}
		}

		if (locOfMax == testLabels[i])
		{
			numTotalTestDigitsCorrect += 1;
			numEachTestDigitCorrect[testLabels[i]] += 1;
		}
	}

	totalClassificationRate = (double)numTotalTestDigitsCorrect / 1000.0;

	for (int i = 0; i < 10; i++)
	{
		digitClassificationRate[i] = (double)numEachTestDigitCorrect[i] / (double)numOfTestExamples[i];
	}
}

void pixelGroupClassifier::printEvaluation()
{
	/*
	Print to screen totalClassificationRate, digitClassificationRate[] for each digit, and confusionMatrix.
	Of course, make it readable and labeled so that the user can understand the results.
	*/

	cout.unsetf(std::ios_base::floatfield);
	cout << setprecision(4);

	cout << "Total classification rate: " << totalClassificationRate << endl << endl;
	cout << "Digit 0 classification rate: " << digitClassificationRate[0] << endl;
	cout << "Digit 1 classification rate: " << digitClassificationRate[1] << endl;
	cout << "Digit 2 classification rate: " << digitClassificationRate[2] << endl;
	cout << "Digit 3 classification rate: " << digitClassificationRate[3] << endl;
	cout << "Digit 4 classification rate: " << digitClassificationRate[4] << endl;
	cout << "Digit 5 classification rate: " << digitClassificationRate[5] << endl;
	cout << "Digit 6 classification rate: " << digitClassificationRate[6] << endl;
	cout << "Digit 7 classification rate: " << digitClassificationRate[7] << endl;
	cout << "Digit 8 classification rate: " << digitClassificationRate[8] << endl;
	cout << "Digit 9 classification rate: " << digitClassificationRate[9] << endl;
	cout << endl;
}
