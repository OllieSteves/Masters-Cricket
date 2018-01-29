#ifndef GETFILE_H
#define GETFILE_H

// Packages
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// get_file function declartion - reads a file, returns a data matrix of strings
std::vector<std::vector<std::string> > get_file(std::string filename);

// read_file function declaration
std::vector<std::vector<int> > read_file(std::string filename);

#endif // GETFILE_H