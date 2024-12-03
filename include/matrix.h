#ifndef DEF_MATRIX
#define DEF_MATRIX

#include <iostream>
#include <sstream>
#include <vector>
#include <assert.h>
#include <cmath>
#include <stdlib.h>

/**
    Every method ends with "const" keyword which means that the object on which the method is called is never modified.
    Instead it will return a new Matrix.

    It is not very efficient, and it could be a lot faster,
    but for the sake of simplicity we will stick to this architecture.
**/

class Matrix
{
public:
    Matrix();
    Matrix(int height, int width);
    Matrix(std::vector<std::vector<double> > const &array);

    Matrix add(Matrix const &m); // addition
    Matrix subtract(Matrix const &m); // subtraction
    Matrix multiply(Matrix const &m); //! hadamard product
    Matrix dot(Matrix const &m) const; // dot product
    Matrix transpose() const; // transposed matrix
    void getMax(int* y,int* x,  double* value) const; // get max value of matrix and its position
    Matrix multiply(double const &value); // scalar multiplication
    Matrix applyFunction(double (*function)(double)); // to apply a function to every element of the matrix
    
    Matrix copy() const;
    int getWidth() const;
    int getHeight() const;
    double get(int i, int j) const;
    void set(int i, int j,double v);
    double sum() const;
    void print(std::ostream &flux) const;
    std::vector<double> getRow(int witchOne) const;
    bool haveAnyNan() const;

private:
    std::vector<std::vector<double>> array;
    int height;
    int width;
};

// overloading operator "<<" to print easily
static std::ostream& operator<<(std::ostream &flux, Matrix const &m)
{
    m.print(flux);
    return flux;
}

#endif
