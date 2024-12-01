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

    auto trainer = DQN();

    Environment2D game; // init environment
    // game.render();
    // game.step(3);
    // game.step(1);
    // usleep(1000000);
    // game.render();
    Policy agent = trainer.train();//uczymy nowego agenta


    cout << "\n\nPlaying game..." << endl;
    usleep(1000000);

    int action;// position = 
    game.reset();
    bool done = false;
    bool help_me = false;
    while(!done){
        game.render(help_me);
        //std::cout << "\r";
        if(!help_me)
            help_me = true;
        Matrix actions = agent.computeOutput({game.getGameRepresentation()});
        actions.getMax( NULL, &action, NULL);
        Observation fb = game.step(action);
        done = fb.done;
        //done = false;
        //position = fb.position;

        usleep(500000);
    }
    game.render();
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