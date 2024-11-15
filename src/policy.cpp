#include "../include/policy.h"

// used to init random weights and biases
double random(double x){
    return (double)(rand() % 10000 + 1)/10000-0.5;
}

// the sigmoid function
double sigmoid(double x){
    return 1/(1+exp(-x));
}

// the derivative of the sigmoid function
double sigmoidePrime(double x){
    return exp(-x)/(pow(1+exp(-x), 2));
}

double stepFunction(double x){
    if(x>0.9){
        return 1.0;
    }
    if(x<0.1){
        return 0.0;
    }
    return x;
}

Policy::Policy(int inputNeuron, int hiddenNeuron, int outputNeuron,double learning_rate){
    learningRate = learning_rate;

    W1 = Matrix(inputNeuron, hiddenNeuron);
    W2 = Matrix(hiddenNeuron, outputNeuron);
    B1 = Matrix(1, hiddenNeuron);
    B2 = Matrix(1, outputNeuron);

    W1 = W1.applyFunction(random);
    W2 = W2.applyFunction(random);
    B1 = B1.applyFunction(random);
    B2 = B2.applyFunction(random);
}

// forward propagation
Matrix Policy::computeOutput(std::vector<double> input){
    X = Matrix({input}); // row matrix
    H = X.dot(W1).add(B1).applyFunction(sigmoid);
    Y = H.dot(W2).add(B2).applyFunction(sigmoid);
    return Y;
}

// back propagation and params update
void Policy::learn(std::vector<double> expectedOutput){
    Y2 = Matrix({expectedOutput}); // row matrix

    // Loss J = 1/2 (expectedOutput - computedOutput)^2
    // Then, we need to calculate the partial derivative of J with respect to W1,W2,B1,B2

    // compute gradients
    dJdB2 = Y2.subtract(Y).multiply(H.dot(W2).add(B2).applyFunction(sigmoidePrime));
    dJdB1 = dJdB2.dot(W2.transpose()).multiply(X.dot(W1).add(B1).applyFunction(sigmoidePrime));
    dJdW2 = H.transpose().dot(dJdB2);
    dJdW1 = X.transpose().dot(dJdB1);

    // update weights and biases
    W1 = W1.subtract(dJdW1.multiply(learningRate));
    W2 = W2.subtract(dJdW2.multiply(learningRate));
    B1 = B1.subtract(dJdB1.multiply(learningRate));
    B2 = B2.subtract(dJdB2.multiply(learningRate));
}