#include "../include/matrix.h"

Matrix::Matrix(){}

Matrix::Matrix(int height, int width)
{
    this->height = height;
    this->width = width;
    this->array = std::vector<std::vector<double> >(height, std::vector<double>(width));
}

Matrix::Matrix(std::vector<std::vector<double> > const &array)
{
    assert(array.size()!=0);
    this->height = array.size();
    this->width = array[0].size();
    this->array = array;
}

Matrix Matrix::multiply(double const &value) const
{
	Matrix result(height, width);
    int i,j;

    for (i=0 ; i<height ; i++)
    {
        for (j=0 ; j<width ; j++)
        {
            result.array[i][j] = array[i][j] * value;
        }
    }

    return result;
}

Matrix Matrix::add(Matrix const &m) const
{
    assert(height==m.height && width==m.width);

    Matrix result(height, width);
    int i,j;

    for (i=0 ; i<height ; i++)
    {
        for (j=0 ; j<width ; j++)
        {
            result.array[i][j] = array[i][j] + m.array[i][j];
        }
    }

    return result;
}

Matrix Matrix::subtract(Matrix const &m) const
{
	assert(height==m.height && width==m.width);

    Matrix result(height, width);
    int i,j;

    for (i=0 ; i<height ; i++)
    {
        for (j=0 ; j<width ; j++)
        {
            result.array[i][j] = array[i][j] - m.array[i][j];
        }
    }

    return result;
}

Matrix Matrix::multiply(Matrix const &m) const
{
    assert(height==m.height && width==m.width);

    Matrix result(height, width);
    int i,j;

    for (i=0 ; i<height ; i++)
    {
        for (j=0 ; j<width ; j++)
        {
            result.array[i][j] = array[i][j] * m.array[i][j];
        }
    }
    return result;
}

Matrix Matrix::dot(Matrix const &m) const
{
    assert(width==m.height);

    int i,j,h, mwidth = m.width;
    double w=0;

    Matrix result(height, mwidth);

    for (i=0 ; i<height ; i++)
    {
        for (j=0 ; j<mwidth ; j++)
        {
            for (h=0 ; h<width ; h++)
            {
                w += array[i][h]*m.array[h][j];
            }
            result.array[i][j] = w;
            w=0;
        }
    }

    return result;
}

Matrix Matrix::transpose() const
{
    Matrix result(width, height);
    int i,j;

    for (i=0 ; i<width ; i++){
        for (j=0 ; j<height ; j++){
            result.array[i][j] = array[j][i];
        }
    }
    return result;
}

Matrix Matrix::applyFunction(double (*function)(double)) const
{
    Matrix result(height, width);
    int i,j;

    for (i=0 ; i<height ; i++)
    {
        for (j=0 ; j<width ; j++){
            result.array[i][j] = (*function)(array[i][j]);
        }
    }

    return result;
}

int Matrix::getWidth() const{
    return width;
}

int Matrix::getHeight() const{
    return height;
}

double Matrix::get(int i, int j) const{
    assert(i>=0 && i<height && j>=0 && j<width);
    return array[i][j];
}

double Matrix::sum() const {
    double sum = 0.0;
    for (int i=0 ; i<height ; i++){
        for (int j=0 ; j<width ; j++){
            sum += array[i][j];
        }
    }
    return sum;
}

// pretty print, taking into account the space between each element of the matrix
void Matrix::print(std::ostream &flux) const
{
    int i,j;
    int maxLength[width] = {};
    std::stringstream ss;

    for (i=0 ; i<height ; i++)
    {
        for (j=0 ; j<width ; j++)
        {
            ss << array[i][j];
            if(maxLength[j] < ss.str().size())
            {
                maxLength[j] = ss.str().size();
            }
            ss.str(std::string());
        }
    }

    for (i=0 ; i<height ; i++)
    {
        for (j=0 ; j<width ; j++)
        {
            flux << array[i][j];
            ss << array[i][j];
            for (int k=0 ; k<maxLength[j]-ss.str().size()+1 ; k++)
            {
                flux << " ";
            }
            ss.str(std::string());
        }
        flux << std::endl;
    }
}
