#ifndef DEF_POLICY
#define DEF_POLICY

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <cmath>
#include <vector>

#include "../include/matrix.h"

class Policy{
    private:
        int hidden_count;
        double learningRate;
    public:
    //  Wagi i bajasy
    std::vector<Matrix> W,B;
    
    //  przechowywanie danych działania
    Matrix X, Y;
    std::vector<Matrix> H;
    std::vector<Matrix> dJdW, dJdB;

    Policy(int inputSize, int hidden_size,int hidden_count, int outputSize,double learning_rate);


    Matrix computeOutput(std::vector<double> input);
    void learn(std::vector<double> expectedOutput,std::vector<double> input,bool update_on_spot = true);
};



#endif