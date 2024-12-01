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
    const int actions[4] = {0,1,2,3}; // left, right
    //const double actionsY[2] = {0, 1}; // left, right
    int steps_done;
    int positionH;
    int positionW;
    int lengthH;
    int lengthW;

    Environment2D(){
        lengthH = 5;
        lengthW = 10;

        positionH = 0;
        positionW = 0;

        steps_done = 0;
    }

    Observation step(int action){
        assert(action >= 0 || action < 4);
        
        steps_done++;
        int punishment = 0;
        if(action==0){
            if(positionH>0){
                positionH--;
            }else{
                punishment += -1;
            }
        }else if(action==1){
            if(positionH<lengthH - 1){
                positionH++;
            }else{
                punishment += -1;
            }
        }else if(action==2){
            if(positionW>0){
                positionW--;
            }else{
                punishment += -1;
            }
        }else if(action==3){
            if(positionW<lengthW - 1){
                positionW++;
            }else{
                punishment += -1;
            }
        }
        if(positionH==lengthH-1 && positionW==lengthW-1)
            return Observation(lengthW + lengthH ,true);//+ distance_to_end_reward() + steps_done_penalty(), true);
        else
            return Observation(punishment,false);//distance_to_end_reward() + steps_done_penalty(), false);
    }

    void eraseLines(int count) {
        if (count > 0) {
            std::cout << "\x1b[2K"; // Delete current line
            // i=1 because we included the first line
            for (int i = 1; i < count; i++) {
                std::cout
                << "\x1b[1A" // Move cursor up one
                << "\x1b[2K"; // Delete the entire line
            }
            std::cout << "\r"; // Resume the cursor at beginning of line
        }
    }

    void render(bool erase = true){
        //Matrix game(lengthH,lengthW);
        //game.set(positionH,positionW,1);
        //game.set(lengthH -1,lengthW - 1,2);
        
        if(erase)
            eraseLines(lengthH+1);
        for(int h = 0;h<lengthH;h++){
            for(int w = 0;w<lengthW;w++){
                int val = 0;
                if(positionH == h && w == positionW){
                    val += 1;
                }
                if(lengthH -1 == h && w == lengthW - 1){
                    val += 2;
                }
                std::cout<<val;
            }
            std::cout<<std::endl;
        }
        //std::cout<<std::endl;
        //std::cout<<game;
        //std::cout<<"rendering, posH:"<<positionH<<" posW:"<<positionW;
    }

    std::vector<double> toGameRepresentation(int h,int w){
        std::vector<double> v(lengthW*lengthH, 0);
        v[h * lengthW + w] += 1;
        v[lengthH * lengthW - 1] += 2;
        return v;
    }

    std::vector<double> getGameRepresentation(){
        //std::cout<<"pre get game";
        std::vector<double> v(lengthH*lengthW, 0);
        v[positionH * lengthW + positionW] += 1;
        v[lengthH * lengthW - 1] += 2;
        //std::cout<<"post get game"<<std::endl;
        return v;
    }

    int length(){// returns a legth of enviroment reprezentation length
        return lengthH*lengthW;
    }

    void reset(){
        positionH = 0;
        positionW = 0;

        steps_done = 0;
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

    // int length(){
    //     return length;
    // }

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
