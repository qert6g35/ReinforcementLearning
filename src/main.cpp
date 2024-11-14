#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <cmath>

#include "../include/DQN.h"

using namespace std;

int main(int argc, char *argv[]){
    srand (time(NULL)); // to generate random weights

    DQN trainer;

    Environment game; // init environment
    Policy agent = trainer.train();//train


    cout << "\n\nPlaying game..." << endl;
    usleep(1000000);

    int action;// position = game.reset();
    bool done = false;
    while(!done){
        game.render();
        std::cout << "\r";

        Matrix actions = agent.computeOutput({game.getGameRepresentation()});
        actions.getMax( NULL, &action, NULL);
        Observation fb = game.step(action);
        done = fb.done;
        //position = fb.position;

        usleep(200000);
    }
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