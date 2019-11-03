// NaiveBayesClassifier.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <string> //to capture user input
#include "naiveBayesDigitClassifier.h"
#include "pixelGroupClassifier.h"

void singlePixels();
void pixelGroups();

int main()
{
	int userAnswer;

	std::cout << "Naive Bayes Digit Classifier\n\n";

	do
	{
		std::cout << "Enter number to choose digit classification type:\n\n"
			<< "1. Single pixels as features\n"
			<< "2. Pixel groups as features\n"
			<< "3. Exit\n\n";

		std::cin >> userAnswer;

		std::cout << std::endl;

		if (userAnswer == 1)
		{
			singlePixels();
		}
		else if (userAnswer == 2)
		{
			pixelGroups();
		}
		else if (userAnswer != 3)
		{
			std::cout << "You entered \"" << userAnswer << "\". Please enter \"1\", \"2\", or \"3\".\n\n";
		}
	} while (userAnswer != 3);

	std::cout << "Naive Bayes Digit Classifier created by Isaiah Gregov and Brett Wilson for CSC 412 - Intro to AI.\n\n";
}

void singlePixels()
{
	std::string userAnswer;

	do
	{
		double smoothingConstant = 0;

		//Retrieve smoothingConstant from user.
		//Parameter check : must be between 0.1 and 10, inclusive.
		do
		{
			std::cout << "Enter smoothing Constant. Must be between 0.1 and 10, inclusive.\n\n";
			std::cin >> smoothingConstant;
			std::cout << std::endl;

			if (smoothingConstant < 0.0 || smoothingConstant > 10.0)
			{
				std::cout << "You entered '" << smoothingConstant << "'. The smoothing constant must be "
					<< "between 0.0 and 10.0, inclusive.\nRe-enter the smoothing constant:\n\n";
			}
		} while (smoothingConstant < 0.0 || smoothingConstant > 10.0);

		naiveBayesDigitClassifier myClassifier(smoothingConstant);
		myClassifier.trainModel();
		myClassifier.testModel();

		std::cout << "Try with a different smoothing constant? (y/n)\n\n";
		std::cin >> userAnswer;
		std::cout << std::endl;
	} while (userAnswer == "y" || userAnswer == "Y");
}

void pixelGroups()
{
	int userAnswer;

	do
	{
		std::cout << "Enter number to choose and run feature set:\n\n"
			<< "1. Disjoint pixel groups of size 2*2\n"
			<< "2. Disjoint pixel groups of size 2*4\n"
			<< "3. Disjoint pixel groups of size 4*2\n"
			<< "4. Disjoint pixel groups of size 4*4\n"
			<< "5. Overlapping pixel groups of size 2*2\n"
			<< "6. Overlapping pixel groups of size 2*4\n"
			<< "7. Overlapping pixel groups of size 4*2\n"
			<< "8. Overlapping pixel groups of size 4*4\n"
			<< "9. Overlapping pixel groups of size 2*3\n"
			<< "10. Overlapping pixel groups of size 3*2\n"
			<< "11. Overlapping pixel groups of size 3*3\n\n";

		std::cin >> userAnswer;

		std::cout << std::endl;

		if (userAnswer < 1 || userAnswer > 11)
		{
			std::cout << "You entered \"" << userAnswer << "\". Please enter a number from 1 to 11.\n\n";
		}
		else
		{
			std::string userAnswerString;

			do
			{
				double smoothingConstant = 0;

				//Retrieve smoothingConstant from user.
				//Parameter check : must be between 0.1 and 10, inclusive.
				do
				{
					std::cout << "Enter smoothing Constant. Must be between 0.1 and 10, inclusive.\n\n";
					std::cin >> smoothingConstant;
					std::cout << std::endl;

					if (smoothingConstant < 0.0 || smoothingConstant > 10.0)
					{
						std::cout << "You entered '" << smoothingConstant << "'. The smoothing constant must be "
							<< "between 0.0 and 10.0, inclusive.\nRe-enter the smoothing constant:\n\n";
					}
				} while (smoothingConstant < 0.0 || smoothingConstant > 10.0);

				pixelGroupClassifier myClassifier(smoothingConstant, userAnswer);
				myClassifier.trainModel();
				myClassifier.testModel();

				std::cout << "Try with a different smoothing constant? (y/n)\n\n";
				std::cin >> userAnswerString;
				std::cout << std::endl;
			} while (userAnswerString == "y" || userAnswerString == "Y");
		}
	} while (userAnswer < 1 || userAnswer > 11);
}