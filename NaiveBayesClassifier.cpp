// NaiveBayesClassifier.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <string> //to capture user input
#include "naiveBayesDigitClassifier.h"

int main()
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
