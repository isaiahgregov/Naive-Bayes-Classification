#pragma once
#include "pixelGroupClassifier.h"
#include <iostream>
#include <fstream>
#include <string> //to capture user input for error-checking
#include <sstream> //stringstream
#include <iomanip> //set precision
#include <math.h> //log and pow functions

using namespace std;

//*******************************************************************************
//--------------Constructors and Destructor-------------------------------------*
//*******************************************************************************

pixelGroupClassifier::pixelGroupClassifier(float fSmoothingConstant, int fFeatureSet)
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
		m = 2;
		n = 2;
		featureSetDisjoint = true;

		break;
	case 2:
		m = 2;
		n = 4;
		featureSetDisjoint = true;

		break;
	case 3:
		m = 4;
		n = 2;
		featureSetDisjoint = true;

		break;
	case 4:
		m = 4;
		n = 4;
		featureSetDisjoint = true;

		break;
	case 5:
		m = 2;
		n = 2;
		featureSetOverlapping = true;

		break;
	case 6:
		m = 2;
		n = 4;
		featureSetOverlapping = true;

		break;
	case 7:
		m = 4;
		n = 2;
		featureSetOverlapping = true;

		break;
	case 8:
		m = 4;
		n = 4;
		featureSetOverlapping = true;

		break;
	case 9:
		m = 2;
		n = 3;
		featureSetOverlapping = true;

		break;
	case 10:
		m = 3;
		n = 2;
		featureSetOverlapping = true;

		break;
	case 11:
		m = 3;
		n = 3;
		featureSetOverlapping = true;

		break;
	}

	numOfFeatureValues = pow(2, (m * n));

	totalClassificationRate = 0;

	for (int i = 0; i < 10; i++)
	{
		priorProbability[i] = 0;
	}

	for (int i = 0; i < 1000; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			posteriorProbability[i][j] = 0;
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

	//Dynamically allocate the rest of the 3D array.
	for (int i = 0; i < 10; i++)
	{
		if (featureSetDisjoint == true)
		{
			pixelGroupProbability[i] = new float** [28 / m];

			for (int j = 0; j < (28 / m); j++)
			{
				pixelGroupProbability[i][j] = new float* [28 / n];

				for (int k = 0; k < (28 / n); k++)
				{
					pixelGroupProbability[i][j][k] = new float [numOfFeatureValues];
				}
			}
		}
		else if (featureSetOverlapping == true)
		{
			pixelGroupProbability[i] = new float** [29 - m];

			for (int j = 0; j < (29 - m); j++)
			{
				pixelGroupProbability[i][j] = new float* [29 - n];

				for (int k = 0; k < (29 - n); k++)
				{
					pixelGroupProbability[i][j][k] = new float[numOfFeatureValues];
				}
			}
		}
	}

	//Assign values to allocated memory.
	for (int i = 0; i < 10; i++)
	{
		if (featureSetDisjoint == true)
		{
			for (int i = 0; i < 10; i++)
				for (int j = 0; j < (28 / m); j++)
					for (int k = 0; k < (28 / n); k++)
						for (int l = 0; l < numOfFeatureValues; l++)
							pixelGroupProbability[i][j][k][l] = 0;
		}
		else if (featureSetOverlapping == true)
		{
			for (int i = 0; i < 10; i++)
				for (int j = 0; j < (29 - m); j++)
					for (int k = 0; k < (29 - n); k++)
						for (int l = 0; l < numOfFeatureValues; l++)
							pixelGroupProbability[i][j][k][l] = 0;
		}
	}
}

pixelGroupClassifier::~pixelGroupClassifier()
{
	//Deallocate memory.
	if (featureSetDisjoint == true)
	{
		for (int i = 0; i < 10; i++)
		{
			for (int j = 0; j < (28 / m); j++)
			{
				for (int k = 0; k < (28 / n); k++)
				{
					delete[] pixelGroupProbability[i][j][k];
					pixelGroupProbability[i][j][k] = NULL;
				}

				delete[] pixelGroupProbability[i][j];
				pixelGroupProbability[i][j] = NULL;
			}

			delete[] pixelGroupProbability[i];
			pixelGroupProbability[i] = NULL;
		}
	}
	else if (featureSetOverlapping == true)
	{
		for (int i = 0; i < 10; i++)
		{
			for (int j = 0; j < (29 - m); j++)
			{
				for (int k = 0; k < (29 - n); k++)
				{
					delete[] pixelGroupProbability[i][j][k];
					pixelGroupProbability[i][j][k] = NULL;
				}

				delete[] pixelGroupProbability[i][j];
				pixelGroupProbability[i][j] = NULL;
			}

			delete[] pixelGroupProbability[i];
			pixelGroupProbability[i] = NULL;
		}
	}

	delete[] pixelGroupProbability;
	pixelGroupProbability = NULL;
}

//*******************************************************************************
//--------------Public Functions------------------------------------------------*
//*******************************************************************************

void pixelGroupClassifier::trainModel()
{
	//A temporary array of twenty-eight (28) 28 char-length strings to update the pixelCountMatrices
	char nextTrainingImage[28][28];

	int nextLabel;

	//An array of ten (10) ... x ... x 16 float-type matrices to keep count of the number of times a pixel group is 
	//“in the foreground” / “counts” for a digit, for each digit. Each element of this array corresponds
	//to each digit from 0 - 9. It is a float or double type for laplace smoothing.
	float**** pixelGroupCountMatrices = new float*** [10];

	//Dynamically allocate the rest of the 3D array.
	for (int i = 0; i < 10; i++)
	{
		if (featureSetDisjoint == true)
		{
			pixelGroupCountMatrices[i] = new float** [28 / m];

			for (int j = 0; j < (28 / m); j++)
			{
				pixelGroupCountMatrices[i][j] = new float* [28 / n];

				for (int k = 0; k < (28 / n); k++)
				{
					pixelGroupCountMatrices[i][j][k] = new float[numOfFeatureValues];
				}
			}
		}
		else if (featureSetOverlapping == true)
		{
			pixelGroupCountMatrices[i] = new float** [29 - m];

			for (int j = 0; j < (29 - m); j++)
			{
				pixelGroupCountMatrices[i][j] = new float* [29 - n];

				for (int k = 0; k < (29 - n); k++)
				{
					pixelGroupCountMatrices[i][j][k] = new float[numOfFeatureValues];
				}
			}
		}
	}

	//Assign values to allocated memory.
	for (int i = 0; i < 10; i++)
	{
		if (featureSetDisjoint == true)
		{
			for (int i = 0; i < 10; i++)
				for (int j = 0; j < (28 / m); j++)
					for (int k = 0; k < (28 / n); k++)
						for (int l = 0; l < numOfFeatureValues; l++)
							pixelGroupCountMatrices[i][j][k][l] = 0;
		}
		else if (featureSetOverlapping == true)
		{
			for (int i = 0; i < 10; i++)
				for (int j = 0; j < (29 - m); j++)
					for (int k = 0; k < (29 - n); k++)
						for (int l = 0; l < numOfFeatureValues; l++)
							pixelGroupCountMatrices[i][j][k][l] = 0;
		}
	}

	//A ten (10) - element array that keeps count of the number of training examples for each digit, which
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
			For each pixel group element that contains one of the m x n char combinations in nextTrainingImage,
			increment the element for pixelGroupCountMatrices[nextLabel][...][...][pixelGroupNumber] by one (1).
		Loop 10 x ... x ... 2^(m * n) times:
			Update pixelGroupProbability for each pixel group counted in pixelGroupCountMatrices for each digit and laplace smooth,
			implementing the conditional probability formula, where P(Gij | class) is updated in pixelProbability:
				P(pixelGroupij | class) = (# of times pixelGroup(i, j) is counted in training examples from this
				class = pixelGroupCountMatrices[digit being classified(determined by the loop number)])
				+ smoothingConstant / (Total # of training examples from this class =
				numOfTrainingExamples[digit being classified(determined by the loop number)] + 2 * smoothingConstant)
		Loop 10 times:
			For each loop, update each digit in priorProbability:
			priorProbability[loop number] = numOfTrainingExamples[loop number] / 5000
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

		//For each pixel group element that contains one of the m x n char combinations in nextTrainingImage,
		//increment the element for pixelGroupCountMatrices[nextLabel][...][...][pixelGroupNumber] by one (1).
		if (featureSetDisjoint == true)
		{
			for (int j = 0; j < (28 / m); j++)
			{
				for (int k = 0; k < (28 / n); k++)
				{
					pixelGroupCountMatrices[nextLabel][j][k][getPixelGroupNumber(nextTrainingImage, (j * m), (k * n))] += 1;
				}
			}
		}
		else if (featureSetOverlapping == true)
		{
			for (int j = 0; j < (29 - m); j++)
			{
				for (int k = 0; k < (29 - n); k++)
				{
					pixelGroupCountMatrices[nextLabel][j][k][getPixelGroupNumber(nextTrainingImage, j, k)] += 1;
				}
			}
		}

		/*
		Loop 10 x ... x ... x 2^(m * n) times:
			Update pixelGroupProbability for each pixel group counted in pixelGroupCountMatrices for each digit and laplace smooth,
			implementing the conditional probability formula, where P(Gij | class) is updated in pixelGroupProbability:

				P(pixelGroupij | class) = (# of times pixel(i, j) is counted in training examples from this
				class = pixelGroupCountMatrices[digit being classified(determined by the loop number)])
				+ smoothingConstant / (Total # of training examples from this class =
				numOfTrainingExamples[digit being classified(determined by the loop number)] + 2 * smoothingConstant)
		*/

		if (featureSetDisjoint == true)
		{
			for (int j = 0; j < 10; j++)
			{
				for (int k = 0; k < (28 / m); k++)
				{
					for (int l = 0; l < (28 / n); l++)
					{
						for (int a = 0; a < numOfFeatureValues; a++)
						{
							pixelGroupProbability[j][k][l][a] = (pixelGroupCountMatrices[j][k][l][a] + smoothingConstant)
								/ (numOfTrainingExamples[j] + numOfFeatureValues * smoothingConstant);
						}
					}
				}
			}
		}
		else if (featureSetOverlapping == true)
		{
			for (int j = 0; j < 10; j++)
			{
				for (int k = 0; k < (29 - m); k++)
				{
					for (int l = 0; l < (29 - n); l++)
					{
						for (int a = 0; a < numOfFeatureValues; a++)
						{
							pixelGroupProbability[j][k][l][a] = (pixelGroupCountMatrices[j][k][l][a] + smoothingConstant)
								/ (numOfTrainingExamples[j] + numOfFeatureValues * smoothingConstant);
						}
					}
				}
			}
		}

		/*
		Loop 10 times:
			Update each digit in priorProbability:
			priorProbability[loop number] = numOfTrainingExamples[loop number] / 5000
		*/

		for (int j = 0; j < 10; j++)
		{
			priorProbability[j] = (float)numOfTrainingExamples[j] / (float)5000;
		}
	}

	//Close data files.
	dataFileInTrainingImages.close();
	dataFileInTrainingLabels.close();

	//Deallocate memory.
	if (featureSetDisjoint == true)
	{
		for (int i = 0; i < 10; i++)
		{
			for (int j = 0; j < (28 / m); j++)
			{
				for (int k = 0; k < (28 / n); k++)
				{
					delete[] pixelGroupCountMatrices[i][j][k];
					pixelGroupCountMatrices[i][j][k] = NULL;
				}

				delete[] pixelGroupCountMatrices[i][j];
				pixelGroupCountMatrices[i][j] = NULL;
			}

			delete[] pixelGroupCountMatrices[i];
			pixelGroupCountMatrices[i] = NULL;
		}
	}
	else if (featureSetOverlapping == true)
	{
		for (int i = 0; i < 10; i++)
		{
			for (int j = 0; j < (29 - m); j++)
			{
				for (int k = 0; k < (29 - n); k++)
				{
					delete[] pixelGroupCountMatrices[i][j][k];
					pixelGroupCountMatrices[i][j][k] = NULL;
				}

				delete[] pixelGroupCountMatrices[i][j];
				pixelGroupCountMatrices[i][j] = NULL;
			}

			delete[] pixelGroupCountMatrices[i];
			pixelGroupCountMatrices[i] = NULL;
		}
	}

	delete[] pixelGroupCountMatrices;
	pixelGroupCountMatrices = NULL;
}

void pixelGroupClassifier::testModel()
{
	//A temporary array of twenty-eight (28) 28 char-length strings to find that image’s posterior probability
	char nextTestImage[28][28];

	int nextLabel;

	/*
		Algorithm:
			Loop 1000 times (testDigitCount):
				Read in next 28 lines from digit data test images file into nextTestImage.
				Read in the next digit from the test labels file into testLabels[testDigitcount].
				numOfTestExamples[testLabels[testDigitCount]] += 1

				Note: Exact implementation following is out of date since it came from Part 1.

				Implement the formula P(class) ∙ P(G1,1 | class) ∙ P(G1,2 | class) ∙ ... ∙ P(G...,... | class) to find posterior probabilities:
				Comment: Change these values to logs as the assignment description says, if underflow occurs
				(and add instead—see assignment description for formula.).
					float pixelProduct = 1;
						Comment: pixelProduct is a temporary value to hold the latter portion of the product of the above formula.
					Loop 10 times (i):
						Loop ... times (j):
							Loop ... times (k):
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

	float maxLog = 0;

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
		Note: Exact implementation following is out of date since it came from Part 1.
		Implement the formula P(class) ∙ P(G1,1 | class) ∙ P(G1,2 | class) ∙ ... ∙ P(G...,... | class) to find posterior probabilities:
		Comment: Change these values to logs as the assignment description says, if underflow occurs
		(and add instead—see assignment description for formula.).
			float pixelProduct = 1;
				Comment: pixelProduct is a temporary value to hold the latter portion of the product of the above formula.
			Loop 10 times (i):
				Loop ... times (j):
					Loop ... times (k):
						If nextTestImage[j, k] == “ “:
							pixelProduct = pixelProduct * (1 - pixelProbability[i, j, k])
						Else if nextTestImage[j, k] == “+“ or “#”:
							pixelProduct = pixelProduct * pixelGroupProbability[i, j, k]
				posteriorProbability[testDigitCount, i] = pixelGroupProbability[i] * pixelProduct

		Note that in the implementation of the probability formula to find posterior probabilities for each class (digit) below,
		I had to add the logs of the probabilities instead of the multiplying the probabilities to avoid underflow;
		then since for a probability (which is a value between 0 and 1) a lower probability increases the negative of
		its log and a higher probability decreases the negative of its logs, I added the negative of the logs and then
		subtracted the total from 1 to find a proportional probability in regard to each class for each class.
		*/

		for (int j = 0; j < 10; j++)
		{
			//pixelProduct is a temporary value to hold the latter portion of the product of the above formula.
			float pixelGroupProduct = 0;

			if (featureSetDisjoint == true)
			{
				for (int k = 0; k < (28 / m); k++)
				{
					for (int l = 0; l < (28 / n); l++)
					{
						//Put here instead of in following for loop to save time.
						int tempPixelGroupNumber = getPixelGroupNumber(nextTestImage, (k * m), (l * n));

						for (int a = 0; a < numOfFeatureValues; a++)
						{
							if (tempPixelGroupNumber != a)
							{
								pixelGroupProduct = pixelGroupProduct - log(1 - pixelGroupProbability[j][k][l][a]);
							}
							else //if (tempPixelGroupNumber == a)
							{
								pixelGroupProduct = pixelGroupProduct - log(pixelGroupProbability[j][k][l][a]);
							}
						}
					}
				}
			}
			else if (featureSetOverlapping == true)
			{
				for (int k = 0; k < (29 - m); k++)
				{
					for (int l = 0; l < (29 - n); l++)
					{
						//Put here instead of in following for loop to save time.
						int tempPixelGroupNumber = getPixelGroupNumber(nextTestImage, k, l);

						for (int a = 0; a < numOfFeatureValues; a++)
						{
							if (tempPixelGroupNumber != a)
							{
								pixelGroupProduct = pixelGroupProduct - log(1 - pixelGroupProbability[j][k][l][a]);
							}
							else //if (tempPixelGroupNumber == a)
							{
								pixelGroupProduct = pixelGroupProduct - log(pixelGroupProbability[j][k][l][a]);
							}
						}
					}
				}
			}

			posteriorProbability[i][j] = pixelGroupProduct - log(priorProbability[j]);

			if (posteriorProbability[i][j] > maxLog)
			{
				maxLog = posteriorProbability[i][j];
			}
		}
	}

	for (int i = 0; i < 1000; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			posteriorProbability[i][j] = posteriorProbability[i][j] / maxLog;
			posteriorProbability[i][j] = 1 - posteriorProbability[i][j];
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
		float max = 0;

		for (int j = 0; j < 10; j++)
		{
			if (posteriorProbability[i][j] > max)
			{
				max = posteriorProbability[i][j];
				locOfMax = j;
			}
		}

		if (locOfMax == testLabels[i])
		{
			numTotalTestDigitsCorrect += 1;
			numEachTestDigitCorrect[testLabels[i]] += 1;
		}
	}

	totalClassificationRate = (float)numTotalTestDigitsCorrect / 1000.0;

	for (int i = 0; i < 10; i++)
	{
		digitClassificationRate[i] = (float)numEachTestDigitCorrect[i] / (float)numOfTestExamples[i];
	}
}

void pixelGroupClassifier::printEvaluation()
{
	/*
	Print to screen totalClassificationRate and digitClassificationRate[] for each digit.
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

//pixeGroupNumber: a unique value corresponding to the combination of pixels that make up the pixel group
int pixelGroupClassifier::getPixelGroupNumber(char nextTrainingImage[28][28], int topLeftRow, int topLeftCol)
{
	string pixelGroupCombination = "";

	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (nextTrainingImage[topLeftRow + 1][topLeftCol + 1] == ' ')
			{
				pixelGroupCombination.append("0");
			}
			else
			{
				pixelGroupCombination.append("1");
			}
		}
	}

	return stoi(pixelGroupCombination, 0, 2);
}
