#ifndef DEF_POLICY
#define DEF_POLICY

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <cmath>

#include "../include/matrix.h"

class Policy{
    public:
    /*
    


    */
    int n_layer = 1;
    Matrix X, W1, H, W2, Y, B1, B2, Y2, dJdB1, dJdB2, dJdW1, dJdW2;
    double learningRate;

    Policy(int inputNeuron, int hiddenNeuron, int outputNeuron,double learning_rate);


    Matrix computeOutput(std::vector<double> input);
    void learn(std::vector<double> expectedOutput);
};



#endif