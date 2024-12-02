#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <cmath>

#include "../include/environment.h"
#include "../include/DQN.h"

using namespace std;

void run_time_tests(){
    int n_samples = 100;
    int samples = 0;
    int instanceH = 8;
    int instanceW = 8;
    int alg_types = 4;
    std::ofstream out_DQN_full;//0
    out_DQN_full.open("DQN_memory_target.csv", std::ios::app);
    std::ofstream out_DQN_mem;//1
    out_DQN_mem.open("DQN_memory.csv", std::ios::app);
    std::ofstream out_DQN_tar;//2
    out_DQN_tar.open("DQN_target.csv", std::ios::app);
    std::ofstream out_DQN;//3
    out_DQN.open("DQN.csv", std::ios::app);
    
    auto trainer = DQN();
    double time = 0;
    for(int iH = 1; iH <= instanceH; iH++){
        for(int iW = 1; iW <= instanceW; iW++){
            trainer.changeGame(iH,iW);
            for(int dqnType = 0; dqnType < alg_types;dqnType++ ){
                switch (dqnType)
                {
                case 0:
                    trainer.use_memory = true;
                    trainer.use_target_agent = true;
                    break;
                case 1:
                    trainer.use_memory = true;
                    trainer.use_target_agent = false;
                    break;
                case 2:
                    trainer.use_memory = false;
                    trainer.use_target_agent = true;
                    break;
                case 3:
                    trainer.use_memory = false;
                    trainer.use_target_agent = false;
                    break;
                default:
                    break;
                }
                for(int s=0;s<samples;s++){
                    trainer.resetAgents();
                    trainer.train(&time);
                }
            }
        }
    }
    //trainer.train(&time);//uczymy nowego agenta
}

void show_how_program_works(){
    srand (time(NULL)); // to generate random weights

    auto trainer = DQN();
    double time = 0;
    Environment2D game = Environment2D(); // init environment
    // game.render();
    // game.step(3);
    // game.step(1);
    // usleep(1000000);
    // game.render();
    Policy agent = trainer.train(&time);//uczymy nowego agenta

    cout << "trained for: " << time << "s" <<endl;

    cout << "\n\nPlaying game..." << endl;
    usleep(1000000);

    game.check_if_good_enougth(agent,true);
}

int main(int argc, char *argv[]){
    srand (time(NULL)); // to generate random weights

    show_how_program_works();

    return 0;
}

/** MAIN
 * 
 *   
 *      
 *TODO Zrozumiec/rozpisać jak tak naprawdę działa zaimplementowany algorytm uczenia aproxymatora tablicy Q   
 *TODO naprawić TODO w DQN
 *TODO      
 *TODO 
 *TODO 
 * 
 * 
 */