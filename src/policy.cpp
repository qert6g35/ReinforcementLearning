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

void Policy::updateParameters(Policy * actual_policy){
    for(int n = 0;n<hidden_count+1;n++){
        W[n] = actual_policy->W[n];
        B[n] = actual_policy->B[n];
    }
    // W.clear();
    // for (auto i : actual_policy.W)
    //     W.push_back(i);
    // B.clear();
    // for (auto i : actual_policy.B)
    //     B.push_back(i);
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

    // for(int i = 0; i<init_threds;i++){
    //     t_H.push_back(std::vector<Matrix>());
    //     for(int n = 0;n<hidden_count+1;n++){
    //         t_H[i].push_back(Matrix());
    //     }
    //     t_X.push_back(Matrix());
    //     t_Y.push_back(Matrix());
    // }
}

Policy::Policy(int inputSize, int hidden_size,int _hidden_count, int outputSize,float learning_rate){
    learningRate = learning_rate;
    hidden_count = _hidden_count;

    int input_size = inputSize;
    int output_size = hidden_size;

    // for(int i = 0; i<init_threds;i++){
    //     t_H.push_back(std::vector<Matrix>());
    //     t_X.push_back(Matrix());
    //     t_Y.push_back(Matrix());
    // }

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
            // for(int i = 0; i<init_threds;i++ ){
            //     t_H[i].push_back(Matrix(hidden_count, 1));
            // }
            H.push_back(Matrix(hidden_count, 1));
        }
    }
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

// // forward propagation
// Matrix Policy::computeOutput_thread(std::vector<float> input,int threadID){
//     t_X[threadID] = Matrix({input});
//     //Z = computeOutput(oldGameRepresentation);
//     for(int n = 0;n <hidden_count + 1;n++){
//         //std::cout<<"just calculated: "<<std::endl;
//         if(n == 0){//! dla pierwszej warstwy
//             t_H[threadID][n] = t_X[threadID].dot(W[n]).add(B[n]).applyFunction(sigmoid); // n = 0
//             //H[n].print(std::cout);
//         }else if(n == hidden_count){//! dla ostatniej warstwy
//             return t_H[threadID][n-1].dot(W[n]).add(B[n]).applyFunction(LeakyReLU); // n = hidden_count
//             //Y.print(std::cout);
//         }else{//! dla każdej innej warstwy
//             t_H[threadID][n] = t_H[threadID][n-1].dot(W[n]).add(B[n]).applyFunction(LeakyReLU);
//             //H[n].print(std::cout);
//         }
//     }
//     return Y;
// }   

// forward propagation
Matrix Policy::computeOutput_fast(std::vector<float> const & input){
    Matrix out = Matrix({input});
    //Z = computeOutput(oldGameRepresentation);
    for(int n = 0;n <hidden_count + 1;n++){
        //std::cout<<"just calculated: "<<std::endl;
        if(n == 0){//! dla pierwszej warstwy
            out = out.dot(W[n]).add(B[n]).applyFunction(sigmoid); // n = 0
            //H[n].print(std::cout);
        }else if(n == hidden_count){//! dla ostatniej warstwy
            return out.dot(W[n]).add(B[n]).applyFunction(LeakyReLU); // n = hidden_count
            //Y.print(std::cout);
        }else{//! dla każdej innej warstwy
            out = out.dot(W[n]).add(B[n]).applyFunction(LeakyReLU);
            //H[n].print(std::cout);
        }
    }
    return Y;
}   

// back propagation and params update
void Policy::learn(float q_correction,int action,std::vector<float> oldGameRepresentation,bool update_weights,float batches_to_add){ // row matrix
    Matrix Y2 = computeOutput(oldGameRepresentation);
    float weigth_of_sample = (float)learningRate/batches_to_add;
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
            dJdW[1].multiply(weigth_of_sample);
            dJdB[1].multiply(weigth_of_sample);
            dJdW[0].multiply(weigth_of_sample);
            dJdB[0].multiply(weigth_of_sample);
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
            dJdW[1].multiply(weigth_of_sample);
            dJdB[1].multiply(weigth_of_sample);
        }
    }
    //std::cout<<"we run"<<batches_to_add<<" batches,  weigth of sample:"<<weigth_of_sample<<"  counted as:"<<learningRate<<"/"<<batches_to_add<<std::endl;
    //int i = 0;
    //while(batches_to_add > i){
    // for(int n = 0;n<hidden_count + 1;n++){
    //     dJdW[n].multiply(weigth_of_sample);
    //     dJdB[n].multiply(weigth_of_sample);
    // }
    if(update_weights)
        change_weights();
}   

void Policy::change_weights(bool clear_derivatives_memory){

    for(int n = 0;n<hidden_count + 1;n++){
        W[n].add(dJdW[n]);
        B[n].add(dJdB[n]);
    }

    if(clear_derivatives_memory){
        clear_weigths_memory();
    }
}

void Policy::change_weights_by_other_policy(Policy * updater){
    for(int n = 0;n<hidden_count + 1;n++){
        W[n].add(updater->dJdW[n]);
        B[n].add(updater->dJdB[n]);
    }
}



void Policy::clear_weigths_memory(){
    //std::cout<<"clear_weigths_memory PRE"<<std::endl;
    dJdW.clear();
    dJdB.clear();
    //std::cout<<"clear_weigths_memory POST"<<std::endl;
}

// back propagation and params update
// void Policy::learn_thread(float q_correction,int action,std::vector<float> oldGameRepresentation,int thread_num,std::mutex& mtxW,std::mutex& mtxB){ // row matrix


//     Matrix D = computeOutput_thread(oldGameRepresentation,thread_num);

//     // for(int n = 0;n <hidden_count + 1;n++){
//     //     //std::cout<<"just calculated: "<<std::endl;
//     //     if(n == 0){//! dla pierwszej warstwy
//     //         t_H[thread_num][n] = t_X[thread_num].dot(W[n]).add(B[n]).applyFunction(sigmoid); // n = 0
//     //         //H[n].print(std::cout);
//     //     }else if(n == hidden_count){//! dla ostatniej warstwy
//     //         D = t_H[thread_num][n-1].dot(W[n]).add(B[n]).applyFunction(LeakyReLU); // n = hidden_count
//     //         //Y.print(std::cout);
//     //     }else{//! dla każdej innej warstwy
//     //         t_H[thread_num][n] = t_H[thread_num][n-1].dot(W[n]).add(B[n]).applyFunction(LeakyReLU);
//     //         //H[n].print(std::cout);
//     //     }
//     // }


//     std::vector<Matrix> dJdW_local, dJdB_local;
//     Matrix Y2 = D.copy();
//     D.set(0,action,q_correction);
//     D.subtract(Y2);
//     for(int n = hidden_count - 1;n>=-1;n--){
//         //std::cout<<"calculate dJdWB n:"<<n<<std::endl;
//         if(n == -1){//! dla pierwszej warstwy
//             D = dJdB_local.front().dot(W[n+2].transpose());
//             dJdB_local.insert(dJdB_local.begin(), D.multiply(t_X[thread_num].dot(W[n+1]).add(B[n+1]).applyFunction(sigmoidePrime)));
//             dJdW_local.insert(dJdW_local.begin(), t_X[thread_num].transpose().dot(dJdB_local.front()));
//         }else if(n == hidden_count - 1){//! dla ostatniej warstwy
//             //std::cout<<Y2<<"|"<<Y<<"|"<<D<<std::endl;
//             //std::cout<<" my multiplie "<<std::endl;
//             //std::cout<<D<<std::endl;
//             //std::cout<<H[n].dot(W[n+1]).add(B[n+1]).applyFunction(LeakyReLUPrime)<<std::endl;
//             dJdB_local.insert(dJdB_local.begin(), D.multiply(t_H[thread_num][n].dot(W[n+1]).add(B[n+1]).applyFunction(LeakyReLUPrime)));
//             //std::cout<<dJdB.front()<<std::endl<<"___"<<std::endl;
//             dJdW_local.insert(dJdW_local.begin(), t_H[thread_num][n].transpose().dot(dJdB_local.front()));
//         }else{//! dla każdej innej warstwy
//             //std::cout<<"1"<<std::endl;
//             D = dJdB_local.front().dot(W[n+2].transpose());
//             //std::cout<<"2"<<std::endl;
//             dJdB_local.insert(dJdB_local.begin(), D.multiply(t_H[thread_num][n].dot(W[n+1]).add(B[n+1]).applyFunction(LeakyReLUPrime)));
//             //std::cout<<"3"<<n<<std::endl;
//             dJdW_local.insert(dJdW_local.begin(), t_H[thread_num][n].transpose().dot(dJdB_local.front()));
//             //std::cout<<"1"<<n<<std::endl;
//         }
//     }
//     int i;
//     mtxW.lock();
//     if(dJdW.size() == 0){
//         for(i = 0;i<dJdW_local.size();i++){
//             dJdW.push_back(dJdW_local[i]);
//         }
//     }else{
//         for(i = 0;i<dJdW_local.size();i++){
//             dJdW[i].add(dJdW_local[i]);
//         }
//     }
//     mtxW.unlock();
//     mtxB.lock();
//     if(dJdB.size() == 0){
//         for(i = 0;i<dJdB_local.size();i++){
//             dJdB.push_back(dJdB_local[i]);
//         }
//     }else{
//         for(i = 0;i<dJdB_local.size();i++){
//             dJdB[i].add(dJdB_local[i]);
//         }
//     }
//     mtxB.unlock();
//     // TODO   dodaj mutex i wpisywanie wartości z lokalnego dJdB i dJdW do globalnych
//     // if(update_on_spot){
//     //     float batches_to_add = (float)dJdW.size()/(float)(hidden_count+1);
//     //     float weigth_of_sample = (float)learningRate/(float)batches_to_add;
//     //     //std::cout<<"we run"<<batches_to_add<<" batches,  weigth of sample:"<<weigth_of_sample<<"  counted as:"<<learningRate<<"/"<<batches_to_add<<std::endl;
//     //     int i = 0;
//     //     while(batches_to_add > i){
//     //         for(int n = 0;n<hidden_count + 1;n++){
//     //             W[n].add(dJdW[n + (hidden_count + 1)*i].multiply(weigth_of_sample));
//     //             B[n].add(dJdB[n + (hidden_count + 1)*i].multiply(weigth_of_sample));
//     //         }
//     //         i++;
//     //     }
//     //     dJdW.clear();
//     //     dJdB.clear();
//     // }

// }  