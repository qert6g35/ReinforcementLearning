#ifndef DEF_POLICY
#define DEF_POLICY

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <cmath>
#include <vector>
#include <thread>
#include <mutex>

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

    //  miejsce na przechowywanie danych przez każdy z wątków
    //std::vector<std::thread> threads;
    std::vector<Matrix> t_X, t_Y;
    std::vector<std::vector<Matrix>> t_H;

public:
    Policy();
    Policy(int inputSize, int hidden_size,int hidden_count, int outputSize,float learning_rate,int init_threds);
    Policy(int n_hidden_count,float n_learningRate,std::vector<Matrix> nW,std::vector<Matrix> nH,std::vector<Matrix> nB);

    Policy copy() const;
    Matrix computeOutput(std::vector<float> input);
    void learn(float q_correction,int action,std::vector<float> oldGameRepresentation);
    void learn_thread(float q_correction,int action,std::vector<float> oldGameRepresentation,int thread_num,std::mutex& mtxW,std::mutex& mtxB);
    std::vector<Matrix> getW() const;
    std::vector<Matrix> getB() const;
    void updateParameters(std::vector<Matrix> W,std::vector<Matrix>B);
    void updateParameters(Policy actual_policy);
    // float activate(float value,int n_layer) const;
    // float activatePrime(float value,int n_layer) const;
};



#endif