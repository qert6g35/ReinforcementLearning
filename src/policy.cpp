#include "../include/policy.h"

//! użyć mniejszej precyzji mniej niż 32Float

// used to init random weights and biases
float random(float x){
    return (float)(rand() % 10000 + 1)/10000-0.5;
}

// the sigmoid function
float sigmoid(float x){
    return 1/(1+exp(-x));
}

// the derivative of the sigmoid function
float sigmoidePrime(float x){
    return exp(-x)/(pow(1+exp(-x), 2));
}

// the ReLU function
float ReLU(float x){
    if(x<0){
        return 0;
    }else{
        return x;
    }
}

// the derivative of the ReLU function
float ReLUPrime(float x){
    if(x<0){
        return 0;
    }else{
        return 1;
    }
}

// the ReLU function
float LeakyReLU(float x){
    if(x<0){
        return 0.1*x;
    }else{
        return x;
    }
}

// the derivative of the ReLU function
float LeakyReLUPrime(float x){
    if(x<0){
        return 0.1;
    }else{
        return 1;
    }
}

// // Tutuaj można podmienić funkcjie aktywacji dla każdej z warstw (MIE ZAPOMNIJ O PRIME!!)
// float Policy::activate(float value,int n_layer) const{
//     if(n_layer == 0){
//         return sigmoid(value);
//     }else if(n_layer != hidden_count){
//         return ReLU(value);
//     }else{
//         return linear(value);
//     }
// }

// std::function<float(float)> Policy::activatePrime(float value,int n_layer) const{
//     if(n_layer == 0){
//         return sigmoidePrime;
//     }else if(n_layer != hidden_count){
//         return ReLUPrime;
//     }else{
//         return linearPrime;
//     }
// }

void Policy::updateParameters(std::vector<Matrix> newW,std::vector<Matrix> newB){
    W = newW;
    B = newB;
}

void Policy::updateParameters(Policy actual_policy){
    W = actual_policy.W;
    B = actual_policy.B;
}

std::vector<Matrix> Policy::getW() const{
    return W;
}

std::vector<Matrix> Policy::getB() const{
    return B;
}
Policy::Policy(){
}

Policy::Policy(int n_hidden_count,float n_learningRate,std::vector<Matrix> nW,std::vector<Matrix> nH,std::vector<Matrix> nB){
    learningRate = n_learningRate;
    hidden_count = n_hidden_count;

    W = nW;
    H = nH;
    B = nB;
}

Policy::Policy(int inputSize, int hidden_size,int _hidden_count, int outputSize,float learning_rate){
    learningRate = learning_rate;
    hidden_count = _hidden_count;

    int input_size = inputSize;
    int output_size = hidden_size;

    for(int n = 0;n<hidden_count+1;n++){
        
        W.push_back(Matrix(input_size, output_size));
        B.push_back(Matrix(1, output_size));

        //W[n].print(std::cout);
        W[n] = W[n].applyFunction(random);
        B[n] = B[n].applyFunction(random);
        //W[n].print(std::cout);

        input_size = hidden_size;
        if(n == hidden_count-1){
            output_size = outputSize;
        }else{
            H.push_back(Matrix(hidden_count, 1));
        }
    // W1 = Matrix(inputNeuron, hiddenNeuron);
    // W2 = Matrix(hiddenNeuron, outputNeuron);
    // B1 = Matrix(1, hiddenNeuron);
    // B2 = Matrix(1, outputNeuron);

    // W1 = W1.applyFunction(random);
    // W2 = W2.applyFunction(random);
    // B1 = B1.applyFunction(random);
    // B2 = B2.applyFunction(random);
    }
    //std::cout<<"We Policy(int n_hidden_count,float n_learningRate,Matrix W,Matrix H)will expect "<<H.size()<<" hidden layers"<<std::endl;
    //std::cout<<"We have "<<W.size()<<" w and b"<<std::endl;
}


Policy Policy::copy() const{
    //std::cout<<"?";
    return Policy(hidden_count,learningRate,W,H,B);
}


// forward propagation
Matrix Policy::computeOutput(std::vector<float> input){
    //std::cout<<"compute output for: "<<std::endl;
    X = Matrix({input}); // row matrix
    //X.print(std::cout);
    for(int n = 0;n <hidden_count + 1;n++){
        //std::cout<<"just calculated: "<<std::endl;
        if(n == 0){//! dla pierwszej warstwy
            H[n] = X.dot(W[n]).add(B[n]).applyFunction(sigmoid); // n = 0
            //H[n].print(std::cout);
        }else if(n == hidden_count){//! dla ostatniej warstwy
            Y = H[n-1].dot(W[n]).add(B[n]).applyFunction(LeakyReLU); // n = hidden_count
            //Y.print(std::cout);
        }else{//! dla każdej innej warstwy
            H[n] = H[n-1].dot(W[n]).add(B[n]).applyFunction(LeakyReLU);
            //H[n].print(std::cout);
        }
    }
    //std::cout<<"return output "<<std::endl;
    //Y.print(std::cout);
    return Y;
}   

// back propagation and params update
void Policy::learn(float q_correction,int action,std::vector<float> oldGameRepresentation,bool update_on_spot){ // row matrix
    Matrix Y2 = computeOutput(oldGameRepresentation);
    //std::cout<<Y2;
    Y2.set(0,action,q_correction);
    // Loss J = 1/2 (expectedOutput - computedOutput)^2
    // Then, we need to calculate the partial derivative of J with respect to W1,W2,B1,B2
    //std::cout<<"start learn hidden:"<<hidden_count<<std::endl;
    Matrix D;
    for(int n = hidden_count - 1;n>=-1;n--){
        //std::cout<<"calculate dJdWB n:"<<n<<std::endl;
        if(n == -1){//! dla pierwszej warstwy
            D = dJdB.front().dot(W[n+2].transpose());
            dJdB.insert(dJdB.begin(), D.multiply(X.dot(W[n+1]).add(B[n+1]).applyFunction(sigmoidePrime)));
            dJdW.insert(dJdW.begin(), X.transpose().dot(dJdB.front()));
        }else if(n == hidden_count - 1){//! dla ostatniej warstwy
            D = Y2.subtract(Y);
            //std::cout<<Y2<<"|"<<Y<<"|"<<D<<std::endl;
            //std::cout<<" my multiplie "<<std::endl;
            //std::cout<<D<<std::endl;
            //std::cout<<H[n].dot(W[n+1]).add(B[n+1]).applyFunction(LeakyReLUPrime)<<std::endl;
            dJdB.insert(dJdB.begin(), D.multiply(H[n].dot(W[n+1]).add(B[n+1]).applyFunction(LeakyReLUPrime)));
            //std::cout<<dJdB.front()<<std::endl<<"___"<<std::endl;
            dJdW.insert(dJdW.begin(), H[n].transpose().dot(dJdB.front()));
        }else{//! dla każdej innej warstwy
            //std::cout<<"1"<<std::endl;
            D = dJdB.front().dot(W[n+2].transpose());
            //std::cout<<"2"<<std::endl;
            dJdB.insert(dJdB.begin(), D.multiply(H[n].dot(W[n+1]).add(B[n+1]).applyFunction(LeakyReLUPrime)));
            //std::cout<<"3"<<n<<std::endl;
            dJdW.insert(dJdW.begin(), H[n].transpose().dot(dJdB.front()));
            //std::cout<<"1"<<n<<std::endl;
        }
    }

    if(update_on_spot){
        int batches_to_add = dJdW.size()/(hidden_count+1);
        //std::cout<<batches_to_add<<"  "<<dJdW.size()<<"/"<<(hidden_count+1);
        int weigth_of_sample = 1.0/(float)batches_to_add;
        int i = 0;
        while(batches_to_add > i){
            for(int n = 0;n<hidden_count + 1;n++){
                W[n] = W[n].add(dJdW[n + (hidden_count + 1)*i].multiply(learningRate*weigth_of_sample));
                B[n] = B[n].add(dJdB[n + (hidden_count + 1)*i].multiply(learningRate*weigth_of_sample));
            }
            i++;
        }
        dJdW.clear();
        dJdB.clear();
    }

        
    // stara sieć
    //nB;
}   // Matrix D2 = Y2.subtract(Y); // błąd drugiej warstwy
    // Matrix e2 = D2.multiply(H.dot(W2).add(B2).applyFunction(sigmoidePrime)); // błąd wewnętrzy neurownów 2. warstwy

    // dJdW2 = H.transpose().dot(e2);// operacja dot wyznaczy nam kombinacje 
    // dJdB2 = e2; // e2 [e,e] . [1,1]^T

    // Matrix D1 = e2.dot(W2.transpose());//  błąd pierwszej warstwy
    // Matrix e1 = D1.multiply(X.dot(W1).add(B1).applyFunction(sigmoidePrime)); // błąd wewnętrzy neurownów 1. warstwy

    // dJdW1 = X.transpose().dot(e1);
    // dJdB1 = e1;

    // update weights and biases
    // W1 = W1.add(dJdW1.multiply(learningRate));
    // W2 = W2.add(dJdW2.multiply(learningRate));
    // B1 = B1.add(dJdB1.multiply(learningRate));
//  nB;
// }