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
    std::ofstream out_DQN_full;
    out_DQN_full.open("DQN_memory_target.csv", std::ios::app);
    std::ofstream out_DQN_mem;
    out_DQN_mem.open("DQN_memory.csv", std::ios::app);
    std::ofstream out_DQN_tar;
    out_DQN_tar.open("DQN_target.csv", std::ios::app);
    std::ofstream out_DQN;
    out_DQN.open("DQN.csv", std::ios::app);
    // for(){

    // }
    auto trainer = DQN();
    double time = 0;
    trainer.train(&time);//uczymy nowego agenta
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