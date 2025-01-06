
#ifndef dqn
#define dqn

#include <stdlib.h>
#include <condition_variable>
#include <chrono>
#include <random>
#include <thread>
#include <functional>
#include <bits/stdc++.h> 

#include "../include/matrix.h"
#include "../include/environment.h"
#include "../include/policy.h"

using namespace std;
const bool show_output = false;
const bool dev_debug_threading = false;

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

    DQNMemoryUnit(){}
};


class DQN{
private:
    Environment2D game; // init environments

    Policy agent;
    Policy target_agent;
    Policy final_agent;

    const int threads_numer = 129;
    std::thread threads[129];
    bool thread_finished_learning[129];
    bool thread_finished_updateing[129];
    std::mutex change_weigths_of_global_agent;
    // std::mutex safe_to_global_dJdB;
    int learning_batch_size;
    std::condition_variable start_threaded_learning;
    std::mutex start_threaded_learning_mtx;
    std::condition_variable finished_threaded_learning;
    std::mutex finished_threaded_learning_mtx;
    std::condition_variable start_threaded_updateing;
    std::mutex start_threaded_updateing_mtx;
    std::condition_variable finished_threaded_updateing;
    std::mutex finished_threaded_updateing_mtx;
    std::vector<DQNMemoryUnit> memory;
    std::chrono::duration<double,std::milli> exec_time;

    int update_local_agent_frequency; //! UWAGA << zmienna odpowiedzialna za częstotliwość updateowania local_agentów przy uczeniu wielowątkowym. 
    bool update_local_agent = false;
    float gamma;
    float eps; // procent określający z jakim prawdopodobieństwem wykonamy ruch losowo
    float epsDecay; // procent maleje TODO
    int target_agent_update_freaquency;
    int target_agent_count_down;
    int n_steps_in_one_go;
    int episode_n;
    float learning_rate;
    const int max_memory_size = 10000;

    bool network_learned = false;
    
    
    bool threads_keep_working = true;
    
public:

    DQN();

    Policy train(double* learning_time,int* steps_done,int* episodes);

    bool collect_memory_step();
    void learn_from_memory(int thread_id);
    void makeDQN_Thread(int thread_idx);

    //* helper/additional functions 
    void showBestChoicesFor(Policy agent);// Function presents what decision agent will choose for each game-state
    DQNMemoryUnit choose_random_from_memory();// chooseing random memorysample

    void resetAgents(int hidden_count = 10,int hidden_size = 10,int threads_number = thread::hardware_concurrency());
    void changeGame(int sizeH,int sizeW);
    bool have_any_nan(Policy agent);

    bool use_memory = true;
    bool use_target_agent = true;
    bool use_threads = true;
    bool make_only_one_learning_steps_ALWAYS = true;
};


#endif
