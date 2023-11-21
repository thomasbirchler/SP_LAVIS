# Import necessary libraries
import numpy as np


def main_function():
    # Create an array of numbers from 0 to 9
    x = np.arange(10)

    # Create a corresponding array of squares
    y = x ** 2

    # Save squared number into a file
    with open('squared_numbers.txt', 'w') as file:
        for i in range(len(x)):
            file.write(f"{x[i]}^2 = {y[i]}\n")


# Check if this script is the main program
if __name__ == "__main__":
    # Call the main function
    main_function()