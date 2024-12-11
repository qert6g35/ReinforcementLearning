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
    float learningRate;

    //  Wagi i bajasy
    std::vector<Matrix> W,B;
    
    //  przechowywanie danych działania
    Matrix X, Y;
    std::vector<Matrix> H;
    std::vector<Matrix> dJdW, dJdB;

public:
    Policy();
    Policy(int inputSize, int hidden_size,int hidden_count, int outputSize,float learning_rate);
    Policy(int n_hidden_count,float n_learningRate,std::vector<Matrix> nW,std::vector<Matrix> nH,std::vector<Matrix> nB);

    Policy copy() const;
    Matrix computeOutput(std::vector<float> input);
    void learn(float q_correction,int action,std::vector<float> oldGameRepresentation, bool update_on_spot = true);
    void learn_thread(float q_correction,int action,std::vector<float> oldGameRepresentation, bool update_on_spot = true);
    std::vector<Matrix> getW() const;
    std::vector<Matrix> getB() const;
    void updateParameters(std::vector<Matrix> W,std::vector<Matrix>B);
    void updateParameters(Policy actual_policy);
    // float activate(float value,int n_layer) const;
    // float activatePrime(float value,int n_layer) const;
};



#endif