// Getfile definition

// Relevant files
#include "getfile.h"

// Function to reads the file and extracts a matrix of strings
std::vector<std::vector<std::string> > get_file(std::string filename)
{
	// Define a test_file and test_line to get the structure of the data (i.e. #rows/columns)
	std::ifstream test_file(filename);
	std::string test_line;
	std::string file_contents;

	// Firstly, we need to determine the number of rows and columns in the data file
	int nrow = 0;
	int ncol = 0;
    
    // Reads the file line by line
	while (std::getline(test_file, test_line))
	{
		// Read the line number by number (each number is seperated by a space)
		std::stringstream line_stream(test_line);
		std::string cell;
		if(nrow == 0)
		{
			while(std::getline(line_stream, cell, ' '))
			{
				++ncol;
			}
		}
		++nrow;
	}
	
	// Now we know we the data is in the following format
	// std::cout << "Rows: " << nrow << ", Columns: " << ncol << std::endl;

	// Now that we know the structure of the data, we can construct an appropriate storage matrix
	std::vector<std::vector<std::string> > data(nrow, std::vector<std::string>(ncol));
	// Fill the data matrix with 0s
	for(int i = 0; i < nrow; i++)
	{
		for(int j = 0; j < ncol; j++)
		{
			data[i][j] = "0";
		}
	}

	// Define a file and line string to explore the data file
	std::ifstream file(filename);
	std::string line;
	
	// Define our index value for populating the data matrix - starting at 0
	int row_index = 0;

	// Read the data file line by line
	while (std::getline(file, line))
	{
		// Get the current line and define a string to be used for the cell values
		std::stringstream line_stream(line);
		std::string cell;

		// Column index value (reset after reading each line)
		int col_index = 0;
		
		// Fill the data matrix with the cell contents
		while (std::getline(line_stream, cell, ' '))
		{
			data[row_index][col_index] = cell;
			++col_index;
		}
		// Now extact the data from the next row
		++row_index;
	}

    // Return the matrix of data in string format
	return(data);
}

// Function which converts our data matrix of strings to numeric data
std::vector<std::vector<int> > read_file(std::string filename)
{
    // Load the data
    std::vector<std::vector<std::string> > string_data = get_file(filename);

    // Get the shape of the data
	int nrow = string_data.size();
    int ncol = string_data[0].size();

	// Create a data matrix to store data in
	// A vector with nrow rows and ncol columns
    std::vector<std::vector<int> > data(nrow, std::vector<int>(ncol));
    std::vector<int> runs(nrow);
    std::vector<int> not_out(nrow);
	
	// Convert from string to float
	// Read row by row
	for (int i = 0; i < nrow; i++)
	{
        // Read column by column within each row
		for (int j = 0; j < ncol; j++)
		{
            std::string value = string_data[i][j];
            std::string::size_type st; 
            data[i][j] = std::stoi(value, &st);
		}
    }

    /* Print some data
    for(int i = 0; i < nrow; i++)
    {
        for(int j = 0; j < ncol; j++)
        {
            std::cout << data[i][j] << " ";
        }
        std::cout << std::endl;
    }
    */

    // Return the numeric data matrix
    return data;
}