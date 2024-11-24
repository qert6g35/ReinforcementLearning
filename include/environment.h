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
    int steps_done;
    int position;
    int length;

    Environment(){
        length = 10;
        position = 0;
        steps_done = 0;
    }

    Observation step(double action){
        steps_done++;
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
            return Observation(length ,true);//+ distance_to_end_reward() + steps_done_penalty(), true);
        else
            return Observation(0,false);//distance_to_end_reward() + steps_done_penalty(), false);
    }

    double distance_to_end_reward(){
        return (double)position / 2;
    }

    double steps_done_penalty(){
        return - (double)steps_done * (1/(double)length/15.0);
    }

    int reset(){
        steps_done = 0;
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


// old problem 
// struct Observation {
//     double reward;
//     bool done;

//     Observation(double reward, bool done){
//         this->reward = reward;
//         this->done = done;
//     }
// };

// struct Environment {
//     const int actionsCount = 2;
//     const double actions[2] = {0, 1}; // left, right
//     int steps_done;
//     int position;
//     int length;

//     Environment(){
//         length = 10;
//         position = 0;
//         steps_done = 0;
//     }

//     Observation step(double action){
//         steps_done++;
//         if(action==actions[0]){
//             if(position>0){
//                 position--;
//             }
//         }

//         if(action==actions[1]){
//             if(position<length){
//                 position++;
//             }
//         }

//         if(position==length)
//             return Observation(length + distance_to_end_reward() + steps_done_penalty(), true);//! originaly was reward = 1
//         else
//             return Observation(distance_to_end_reward() + steps_done_penalty(), false);//! originaly was reward = 0
//     }

//     double distance_to_end_reward(){
//         return (double)position / 2;
//     }

//     double steps_done_penalty(){
//         return - (double)steps_done * (1/(double)length/2);
//     }

//     int reset(){
//         steps_done = 0;
//         position = 0;
//         return 0;
//     }

//     void render(){
//         std::cout << "[";
//         for(int i=0 ; i<length ; i++){
//             std::cout << (i==position?"X":"_");
//         }
//         std::cout << "]" << std::flush;
//     }

//     std::vector<double> toGameRepresentation(int n, int max){
//         std::vector<double> v(max, 0);
//         v[n] = 1;
//         return v;
//     }

//     std::vector<double> getGameRepresentation(){
//         std::vector<double> v(length, 0);
//         v[position] = 1;
//         return v;
//     }
// };
