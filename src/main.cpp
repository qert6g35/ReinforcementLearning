#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <cmath>

#include "../include/environment.h"
#include "../include/DQN.h"

using namespace std;

void testSingleExaple(){
    int n_samples = 50;
    int instance_max_size = 9;

    std::ofstream writeHere;
    writeHere.open("DQN_extra.csv", std::ios::app);
    
    auto trainer = DQN();
    double time = 0;
    int steps = 0;
    for(int iH = 9; iH <= instance_max_size; iH++){
        for(int iW = 2; iW <= iH; iW++){
            trainer.changeGame(iH,iW);
            cout<<"start sampling for heigth:"<<iH<<", width:"<<iW<<endl;
            for(int s=0;s<n_samples;s++){
                    trainer.resetAgents();
                    trainer.train(&time,&steps);
                    writeHere<<iH<<","<<iW<<","<<time<<","<<steps<<endl;
                }   
        }
    }
    //trainer.train(&time);//uczymy nowego agenta
}

void run_time_tests(int startingH = 2, int startingW = 2){
    int n_samples = 50;
    int instance_max_size = 9;
    int alg_types = 4;

    std::ofstream writeHere;
    writeHere.open("DQN_memory_target.csv", std::ios::app);
    
    auto trainer = DQN();
    double time = 0;
    int steps = 0;
    for(int iH = startingH; iH <= instance_max_size; iH++){
        for(int iW = startingW; iW <= iH; iW++){
            trainer.changeGame(iH,iW);
            cout<<"start sampling for heigth:"<<iH<<", width:"<<iW<<endl;
            for(int dqnType = 0; dqnType < alg_types;dqnType++ ){
                switch (dqnType)
                {
                case 0:
                    //cout<<"start sampling for heigth:"<<iH<<", width:"<<iW<<", agl full_DQN"<<endl;
                    writeHere.close();
                    writeHere.open("DQN_memory_target.csv", std::ios::app);
                    trainer.use_memory = true;
                    trainer.use_target_agent = true;
                    break;
                case 1:
                    //cout<<"start sampling for heigth:"<<iH<<", width:"<<iW<<", agl memory_DQN"<<endl;
                    writeHere.close();
                    writeHere.open("DQN_memory.csv", std::ios::app);
                    trainer.use_memory = true;
                    trainer.use_target_agent = false;
                    break;
                case 2:
                    //cout<<"start sampling for heigth:"<<iH<<", width:"<<iW<<", agl target_DQN"<<endl;
                    writeHere.close();
                    writeHere.open("DQN_target.csv", std::ios::app);
                    trainer.use_memory = false;
                    trainer.use_target_agent = true;
                    break;
                case 3:
                    //cout<<"start sampling for heigth:"<<iH<<", width:"<<iW<<", agl simple_DQN"<<endl;
                    writeHere.close();
                    writeHere.open("DQN.csv", std::ios::app);
                    trainer.use_memory = false;
                    trainer.use_target_agent = false;
                    break;
                default:
                    break;
                }
                for(int s=0;s<n_samples;s++){
                    trainer.resetAgents();
                    trainer.train(&time,&steps);
                    writeHere<<iH<<","<<iW<<","<<time<<","<<steps<<endl;
                }
                //cout<<"finished sampling"<<endl<<endl;
            }   
        }
    }
    //trainer.train(&time);//uczymy nowego agenta
}

void show_how_program_works(){
    srand (time(NULL)); // to generate random weights

    auto trainer = DQN();
    double time = 0;
    int steps = 0;
    Environment2D game = Environment2D(); // init environment
    // game.render();
    // game.step(3);
    // game.step(1);
    // usleep(1000000);
    // game.render();
    Policy agent = trainer.train(&time,&steps);//uczymy nowego agenta

    cout << "trained for: " << time << "s" << " in steps:"<<steps<<endl;

    cout << "\n\nPlaying game..." << endl;
    usleep(100000);

    game.check_if_good_enougth(agent,true);
}

int main(int argc, char *argv[]){
    //srand (time(NULL)); // to generate random weights

    // finish_time_tests(8);
    // run_time_tests(9,8);
    testSingleExaple();
    
    return 0;
}

/** MAIN
 * //! zastanów się, czy nie chcemy użyć np TensorFlow czy innej biblioteki do zastąpienia Polityki
 *   
 *   drobno vs gruboziarnistość
 *TODO w drobno:
 *TODO 
 *TODO 
 *TODO w grubo: wątki vs procesy   
 *TODO        1) rozproszoność (procesy) vs 2) równoległość (threads) w ramach efektywności
 *TODO  
 *TODO      sprawdzić czas generacji nowych sieci
 *TODO 
 *TODO 
 * 
 * 
 */