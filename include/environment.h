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

struct Environment2D
{
    const int actionsCount = 4;
    const double actionsX[2] = {0, 1}; // left, right
    const double actionsY[2] = {0, 1}; // left, right
    int steps_done;
    int positionH;
    int positionW;
    int lengthH;
    int lengthW;

    Environment2D(){
        lengthH = 10;
        lengthW = 10;

        positionH = 0;
        positionW = 0;

        steps_done = 0;
    }

    Observation step(int actionH,int actionW){
        assert(actionH == 0 || actionH == 1);
        assert(actionW == 0 || actionW == 1);
        
        steps_done++;
        int punishment = 0;
        if(actionH==0){
            if(positionH>0){
                positionH--;
            }else{
                punishment += -1;
            }
        }else{
            if(positionH<lengthH - 1){
                positionH++;
            }
        }
        if(actionW==0){
            if(positionW>0){
                positionW--;
            }else{
                punishment += -1;
            }
        }else{
            if(positionW<lengthW){
                positionW++;
            }
        }
        if(positionH==lengthH-1 && positionW==lengthW-1)
            return Observation((lengthW + lengthH)/2 ,true);//+ distance_to_end_reward() + steps_done_penalty(), true);
        else
            return Observation(punishment,false);//distance_to_end_reward() + steps_done_penalty(), false);
    }

    void render(){
        Matrix game(lengthH,lengthW);
        game.set(positionH,positionW,1);
        game.set(lengthH -1,lengthW - 1,2);
        std::cout<<game;
    }

    std::vector<double> toGameRepresentation(int h,int w, int side_len){
        std::vector<double> v(side_len*side_len, 0);
        v[h * side_len + w] += 1;
        v[side_len * side_len - 1] += 2;
        return v;
    }

    std::vector<double> getGameRepresentation(){
        std::vector<double> v(lengthH*lengthW, 0);
        v[positionH * lengthW + lengthW] += 1;
        v[lengthH * lengthW - 1] += 2;
        return v;
    }
};


struct Environment1D {
    const int actionsCount = 2;
    const double actions[2] = {0, 1}; // left, right
    int steps_done;
    int position;
    int length;

    Environment1D(){
        length = 10;
        position = 0;
        steps_done = 0;
    }

    Observation step(double action){
        assert(action == 0 || action == 1);
        steps_done++;
        int punishment = 0;
        if(action==0){
            if(position>0){
                position--;
            }else{
                punishment = -1;
            }
        }else{
            if(position<length-1){
                position++;
            }
        }


        if(position==length-1)
            return Observation(length ,true);//+ distance_to_end_reward() + steps_done_penalty(), true);
        else
            return Observation(punishment,false);//distance_to_end_reward() + steps_done_penalty(), false);

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
