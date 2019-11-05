#pragma once
#include "naiveBayesDigitClassifier.h"
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

naiveBayesDigitClassifier::naiveBayesDigitClassifier(float fSmoothingConstant)
{
	smoothingConstant = fSmoothingConstant;
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

	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			confusionMatrix[i][j] = 0;
		}
	}
}

naiveBayesDigitClassifier::~naiveBayesDigitClassifier()
{
}

//*******************************************************************************
//--------------Public Functions------------------------------------------------*
//*******************************************************************************

void naiveBayesDigitClassifier::trainModel()
{
	//A temporary array of twenty - eight(28) 28 - char - length strings to update the pixelCountMatrices
	char nextTrainingImage[28][28];

	int nextLabel;

	//An array of ten(10) 28 x 28 float - type matrices to keep count of the number of times a pixel is 
	//“in the foreground” / “counts” for a digit, for each digit.Each element of this array corresponds
	//to each digit from 0 - 9. It is a float or double type for laplace smoothing.
	float pixelCountMatrices[10][28][28];

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
			priorProbability[j] = (float)numOfTrainingExamples[j] / (float)5000;
		}
	}

	//Close data files.
	dataFileInTrainingImages.close();
	dataFileInTrainingLabels.close();
}

void naiveBayesDigitClassifier::testModel()
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
					float pixelProduct = 1;
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
		Implement the formula P(class) ∙ P(f1,1 | class) ∙ P(f1,2 | class) ∙ ... ∙ P(f28,28 | class) to find posterior probabilities:
		Comment: Change these values to logs as the assignment description says, if underflow occurs
		(and add instead—see assignment description for formula.).
			float pixelProduct = 1;
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
			float pixelProduct = 0;

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

			testDigitProbability[i][j] = pixelProduct - log(priorProbability[j]);

			if (testDigitProbability[i][j] > maxLog)
			{
				maxLog = testDigitProbability[i][j];
			}
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

void naiveBayesDigitClassifier::evaluateModel()
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

		confusionMatrix[testLabels[i]][locOfMax] += 1;
	}
	
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			confusionMatrix[i][j] = confusionMatrix[i][j] / (float)numOfTestExamples[i];
		}
	}

	totalClassificationRate = (float)numTotalTestDigitsCorrect / 1000.0;

	for (int i = 0; i < 10; i++)
	{
		digitClassificationRate[i] = (float)numEachTestDigitCorrect[i] / (float)numOfTestExamples[i];
	}
}

void naiveBayesDigitClassifier::printEvaluation()
{
	printClassificationRateAndConfusionMatrix();
	printFeatureLikelihoodsAndOddsRatios();
}

void naiveBayesDigitClassifier::printClassificationRateAndConfusionMatrix()
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

	cout << "Confusion matrix, where the entry in row r and column c is the percentage of\n"
		<< "test images from class r that are classified as class c:\n\n";

	cout << setprecision(2) << fixed;

	cout << "\t\t\t\t      Digits\n\t\t  0    1    2    3    4    5    6    7    8    9\n"
		<< "\t\t" << string(50, '-') << endl;

	for (int i = 0; i < 10; i++)
	{
		cout << "Digit " << i << ":\t";

		for (int j = 0; j < 10; j++)
		{
			cout << confusionMatrix[i][j] << " ";
		}

		cout << endl;
	}

	cout << endl << endl << endl;

	/*
	For each digit class, print to screen the test examples from that class that have the
	highest and the lowest posterior probabilities according to the classifier:
	Iterate through testDigitProbability to find the highest and lowest posterior
	probability for each digit. Ties don’t matter but probably are unlikely; the next one
	would just be picked in place of the previous one.
	*/

	float testDigitMaxProbability[10];
	float testDigitMinProbability[10];

	for (int i = 0; i < 10; i++)
	{
		testDigitMaxProbability[i] = 0.0;
		testDigitMinProbability[i] = 1.0;
	}

	int testDigitMaxProbabilityLoc[10];
	int testDigitMinProbabilityLoc[10];
	
	/*
	Loop 10 times(i) :
		Loop 1000 times(j) :
			If testDigitProbability[j][i] > testDigitMaxProbability[i]:
				testDigitMaxProbability[i] = testDigitProbability[j][i]
				testDigitMaxProbabilityLoc[i] = j
			If testDigitProbability[j][i] < testDigitMinProbability[i] :
				testDigitMinProbability[i] = testDigitProbability[j][i]
				testDigitMinProbabilityLoc[i] = j
	Loop 10 times(i):
		Note: The following steps were modified to make the images with the highest and lowest probabilities appear
		next to each other instead of vertically above and below each other, but at least, if you know this, the same
		principle applies: to output the images with the highest and lowest probability for the user to see.
		Open test images file.
		Print “Image with highest probability for digit “ + i + “ : \n\n”
		Read in / skip past the next 28 * testDigitMaxProbabilityLoc[i].Then print the next 28 lines.
		Close test images file and open it again(or bring it back to the beginning somehow—depending on the programming language…)
		Print “Image with lowest probability for digit “ + i + “:\n\n”
		Read in / skip past the next 28 * testDigitMinProbabilityLoc[i].Then print the next 28 lines.
		Close test images file.
	*/

	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 1000; j++)
		{
			if (testDigitProbability[j][i] > testDigitMaxProbability[i])
			{
				testDigitMaxProbability[i] = testDigitProbability[j][i];
				testDigitMaxProbabilityLoc[i] = j;
			}
			else if (testDigitProbability[j][i] < testDigitMinProbability[i])
			{
				testDigitMinProbability[i] = testDigitProbability[j][i];
				testDigitMinProbabilityLoc[i] = j;
			}
		}
	}

	cout << "Test examples from each class that have the highest and the lowest posterior probabilities:\n"
		<< string(100, '-') << endl << endl;

	for (int i = 0; i < 10; i++)
	{
		//Open data file.
		ifstream dataFileInTestImagesForHighestProb;
		ifstream dataFileInTestImagesForLowestProb;
		dataFileInTestImagesForHighestProb.open("testimages");
		dataFileInTestImagesForLowestProb.open("testimages");

		//Strings for processing input from data file.
		string nextLine1, nextLine2;

		cout << "Image with highest probability for digit " << i
			<< ":\t\tImage with lowest probability for digit " << i << ":\n"
			<< "Probability: "
			<< testDigitProbability[testDigitMaxProbabilityLoc[i]][i]
			<< "\t\t\t\t\tProbability: "
			<< testDigitProbability[testDigitMinProbabilityLoc[i]][i];

		for (int j = 0; j < (28 * testDigitMaxProbabilityLoc[i]); j++)
		{
			getline(dataFileInTestImagesForHighestProb, nextLine1);
		}

		for (int j = 0; j < (28 * testDigitMinProbabilityLoc[i]); j++)
		{
			getline(dataFileInTestImagesForLowestProb, nextLine2);
		}

		for (int j = 0; j < 28; j++)
		{
			getline(dataFileInTestImagesForHighestProb, nextLine1);
			getline(dataFileInTestImagesForLowestProb, nextLine2);

			cout << nextLine1 << "\t\t\t\t\t" << nextLine2 << endl;
		}

		//Close data files.
		dataFileInTestImagesForHighestProb.close();
		dataFileInTestImagesForLowestProb.close();
	}

	cout << endl << endl << endl;
}

void naiveBayesDigitClassifier::printFeatureLikelihoodsAndOddsRatios()
{
	float pixelOdds[10][10][28][28];

	//The odds that a pixel belongs to one or the other class:
	//odds(Fij = 1, c1, c2) = P(Fij = 1 | c1) / P(Fij = 1 | c2)

	int max1Row;
	int max1Col;
	int max2Row;
	int max2Col;
	int max3Row;
	int max3Col;
	int max4Row;
	int max4Col;
	float confusionMax1 = 0.0;
	float confusionMax2 = 0.0;
	float confusionMax3 = 0.0;
	float confusionMax4 = 0.0;

	//Find four pairs of digits that have the highest confusion rates according to the confusion matrix.

	/*
	Loop 4 times(i) :
		Loop 10 times(j) :
			Loop 10 times(k) :
				If i = 1 :
					If confusionMatrix[j][k] > confusionMax1:
						confusionMax1 = confusionMatrix[j][k]
						max1Row = j
						max1Col = k
				Else if i = 2 :
					If confusionMatrix[j][k] > confusionMax2and < confusionMax1:
						confusionMax2 = confusionMatrix[j][k]
						max2Row = j
						max2Col = k
				Else if i = 3 :
					If confusionMatrix[j][k] > confusionMax3and < confusionMax2:
						confusionMax3 = confusionMatrix[j][k]
						max3Row = j
						max3Col = k
				Else if i = 4 :
					If confusionMatrix[j][k] > confusionMax4and < confusionMax2:
						confusionMax4 = confusionMatrix[j][k]
						max4Row = j
						max4Col = k
	*/

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			for (int k = 0; k < 10; k++)
			{
				switch (i)
				{
					case 0:
						if (confusionMatrix[j][k] > confusionMax1 && j != k)
						{
							confusionMax1 = confusionMatrix[j][k];
							max1Row = j;
							max1Col = k;
						}
						
						break;
					case 1:
						if (confusionMatrix[j][k] > confusionMax2 && confusionMatrix[j][k] < confusionMax1 && j != k)
						{
							confusionMax2 = confusionMatrix[j][k];
							max2Row = j;
							max2Col = k;
						}

						break;
					case 2:
						if (confusionMatrix[j][k] > confusionMax3 && confusionMatrix[j][k] < confusionMax2 && j != k)
						{
							confusionMax3 = confusionMatrix[j][k];
							max3Row = j;
							max3Col = k;
						}

						break;
					case 3:
						if (confusionMatrix[j][k] > confusionMax4 && confusionMatrix[j][k] < confusionMax3 && j != k)
						{
							confusionMax4 = confusionMatrix[j][k];
							max4Row = j;
							max4Col = k;
						}

						break;
				}
			}
		}
	}

	//Find pixelOdds.

	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			for (int k = 0; k < 28; k++)
			{
				for (int l = 0; l < 28; l++)
				{
					pixelOdds[i][j][k][l] = pixelProbability[i][k][l] / pixelProbability[j][k][l];
				}
			}
		}
	}

	/*
	For each pair, display the maps of feature likelihoods for both classes as well as the odds
	ratio for the two classes. If you cannot do a graphical display, you can display the maps in
	ASCII format using some coding scheme of your choice. For example, for the odds ratio map,
	you can use '+' to denote features with positive log odds, ' ' for features with log odds
	close to 1, and '-' for features with negative log odds.
	*/

	displayFeatureLikelihoodsAndOddsRatiosMaps(pixelOdds, max1Row, max1Col);
	displayFeatureLikelihoodsAndOddsRatiosMaps(pixelOdds, max2Row, max2Col);
	displayFeatureLikelihoodsAndOddsRatiosMaps(pixelOdds, max3Row, max3Col);
	displayFeatureLikelihoodsAndOddsRatiosMaps(pixelOdds, max4Row, max4Col);
}

void naiveBayesDigitClassifier::displayFeatureLikelihoodsAndOddsRatiosMaps(float pixelOdds[10][10][28][28], int maxRow, int maxCol)
{
	cout << "Feature likelihood for digit " << maxRow << ":\t\t"
		<< "Feature likelihood for digit " << maxCol << ":\t\t"
		<< "Odds ratios for digits " << maxRow << " (+) and " << maxCol << " (-):\n\n";

	for (int i = 0; i < 28; i++)
	{
		for (int j = 0; j < 28; j++)
		{
			if (pixelProbability[maxRow][i][j] > 0.65)
			{
				cout << "+";
			}
			else if (pixelProbability[maxRow][i][j] < 0.25)
			{
				cout << "-";
			}
			else
			{
				cout << " ";
			}
		}

		cout << "\t\t";

		for (int j = 0; j < 28; j++)
		{
			if (pixelProbability[maxCol][i][j] > 0.65)
			{
				cout << "+";
			}
			else if (pixelProbability[maxCol][i][j] < 0.25)
			{
				cout << "-";
			}
			else
			{
				cout << " ";
			}
		}

		cout << "\t\t";

		for (int j = 0; j < 28; j++)
		{
			if (pixelOdds[maxRow][maxCol][i][j] > 1.4)
			{
				cout << "+";
			}
			else if (pixelOdds[maxRow][maxCol][i][j] < 0.6)
			{
				cout << "-";
			}
			else
			{
				cout << " ";
			}
		}

		cout << endl;
	}

	cout << endl << endl << endl;
}