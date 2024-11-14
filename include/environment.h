#ifndef ENVIRONMENT
#define ENVIRONMENT

#include <string>
#include <iostream>

struct Observation {
    double reward;
    bool done;

    Observation(double reward, bool done){
        this->reward = reward;
        this->done = done;
    }
};

struct Environment {
    const int actionsCount = 2;
    const double actions[2] = {0, 1}; // left, right
    int position;
    int length;

    Environment(){
        length = 10;
        position = 0;
    }

    Observation step(double action){
        if(action==actions[0]){
            if(position>0){
                position--;
            }
        }

        if(action==actions[1]){
            if(position<length){
                position++;
            }
        }

        if(position==length)
            return Observation(1, true);
        else
            return Observation(0, false);
    }

    int reset(){
        position = 0;
        return 0;
    }

    void render(){
        std::cout << "[";
        for(int i=0 ; i<length ; i++){
            std::cout << (i==position?"X":"_");
        }
        std::cout << "]" << std::flush;
    }

    std::vector<double> toGameRepresentation(int n, int max){
        std::vector<double> v(max, 0);
        v[n] = 1;
        return v;
    }

    std::vector<double> getGameRepresentation(){
        std::vector<double> v(length, 0);
        v[position] = 1;
        return v;
    }
};

#endif
