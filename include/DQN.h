#ifndef dqn
#define dqn

#include <stdlib.h>
//#include <time.h>
#include <chrono>
#include <random>

#include "../include/matrix.h"
#include "../include/environment.h"
#include "../include/policy.h"

using namespace std;
const bool show_output = true;

struct DQNMemoryUnit
{
    std::vector<float> game = std::vector<float>();
    std::vector<float> game_next = std::vector<float>();
    int action;
    float reward; 
    bool done;

    DQNMemoryUnit(std::vector<float> ngame,std::vector<float> ngame_next,int naction,float nreward,bool isDone){
        game = ngame;
        game_next = ngame_next;
        action = naction;
        reward = nreward;
    }

    DQNMemoryUnit(){
        
    }
};


class DQN{
private:
    Environment2D game; // init environments

    Policy agent;
    Policy target_agent;

    std::vector<DQNMemoryUnit> memory;
    std::chrono::duration<double,std::milli> exec_time;

    float gamma;
    float eps; // procent określający z jakim prawdopodobieństwem wykonamy ruch losowo
    float epsDecay; // procent maleje TODO
    int target_agent_update_freaquency;
    int target_agent_count_down;
    int n_steps_in_one_go;
    int episode_n;
    float learning_rate;
    int learning_batch_size;

public:

    DQN();

    Policy train(double* learning_time,int* steps_done,int* episodes);

    bool collect_memory_step();
    void learn_from_memory(bool update_on_spot = true);

    //* helper/additional functions 
    void showBestChoicesFor(Policy agent);// Function presents what decision agent will choose for each game-state
    DQNMemoryUnit choose_random_from_memory();// chooseing random memorysample

    void resetAgents(int hidden_count = 8,int hidden_size = 10);
    void changeGame(int sizeH,int sizeW);

    bool use_memory = true;
    bool use_target_agent = true;
};


#endif
/*







eksplodujący gradient, wartości zaczynają lecieć w nieskończoność + lub - 
agent -2.34e+34 -9.606e+31 i leci const correction -81 action 0 
agent  -3  i - 40                         
agent -0.008 i -0.0917 corection const = -1 na action 0

tutaj zawsze idziemy w lewo, mimo że epsilon jest srestartowany, na każdeym resecie jest szansa że wartości zwaracane przez agenta będą jeszcze gorsze
w momencie resetu (kiedy zablokowany alg zreseruje wsp. eksploracji to na zaminę pogarsza ) 



po zdjęciu blokady na MAX ujemną wartość (-100 było najmniejszą możliwą wartością w getMax)

agent się poddał i wyrzuca -nan -nan













*/