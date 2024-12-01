#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <cmath>

#include "../include/environment.h"
#include "../include/policy.h"
#include "../include/DQN.h"

using namespace std;

int main(int argc, char *argv[]){
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