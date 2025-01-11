
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

    std::ofstream thread_times_file;   
    static const int threads_numer = 129;
    std::thread threads[threads_numer];
    std::chrono::_V2::system_clock::time_point start_learning_time;
    long double learning_times[2][threads_numer]; 
    // times[0][cpu] - start time 
    // times[1][cpu] - stop  time 
    bool thread_finished_learning[threads_numer];
    bool thread_finished_updateing[threads_numer];
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

    Policy get_agent(){
        return agent;
    }

    void set_agent(Policy agent_to_set){
        this->agent = agent_to_set;
    }

    //* helper/additional functions 
    void showBestChoicesFor(Policy agent);// Function presents what decision agent will choose for each game-state
    DQNMemoryUnit choose_random_from_memory();// chooseing random memorysample

    void resetAgents(int hidden_count = 10,int hidden_size = 10,int threads_number = thread::hardware_concurrency());
    void changeGame(int sizeH,int sizeW);
    bool have_any_nan(Policy agent);

    void collect_time(bool start_else_end,int thread_id);
    void safe_data_to_file(bool is_update_times);

    int folder_to_safe_to = 0;
    bool use_memory = true;
    bool use_target_agent = true;
    bool use_threads = true;
    bool make_only_one_learning_steps_ALWAYS = true;
};


#endif
