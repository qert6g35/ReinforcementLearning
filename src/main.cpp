#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <cmath>

#include "../include/matrix.h"
#include "../include/environment.h"
#include "../include/policy.h"

using namespace std;


std::vector<double> toCategorical(int n, int max){
    std::vector<double> v(max, 0);
    v[n] = 1;
    return v;
}

void getMax(const Matrix& row, int* y, int* x, double* value){
    double max=0;
    int ii=0, jj=0;
    for(int i=0 ; i<row.getHeight() ; i++){
        for(int j=0 ; j<row.getWidth() ; j++){
            if(max<row.get(i, j)){
                max = row.get(i, j);
                ii = i;
                jj = j;
            }
        }
    }

    if(x!=NULL){
        *x = jj;
    }
    if(y!=NULL){
        *y = ii;
    }
    if(value!=NULL){
        *value = max;
    }
}

int main(int argc, char *argv[]){
    srand (time(NULL)); // to generate random weights
    Environment game; // init environment
    Policy agent(game.length, 10, game.actionsCount, 0.1);
    // init network
    // input : game state (ex: [1,0,0,0,0,0,0,0,0,0] when agent is at the first position)
    // output : action to take
    //init(game.length, 10, game.actionsCount, 0.1);

    // q learning var
    double gamma = 0.8;
    double eps = 1.0;
    double epsDecay = 0.95;

    // train
    int episode = 100;
    for (int i=0 ; i<episode ; i++){
        cout << "Episode " << i+1 << "/" << episode << "\t";
        int steps=0, maxIndex=0, action=0, position=0;
        double qsa=0, max=0;
        bool done=false;

        position = game.reset();

        while(!done && steps<300){
            steps++;

            // take random actions sometimes to allow game exploration
            if(((double) rand() / RAND_MAX) < eps){
                action = rand()%game.actionsCount;
            }
            else{
                Matrix actions = agent.computeOutput({toCategorical(position, game.length)});
                getMax(actions, NULL, &action, NULL);
            }

            // take action
            Observation fb = game.step(action);

            // get max action in next state
            Matrix nextActions = agent.computeOutput({toCategorical(fb.position, game.length)});
            getMax(nextActions, NULL, &maxIndex, &max);
            qsa = fb.reward + gamma*max;

            // update network
            std::vector<double> in(game.actionsCount);
            for(int i=0 ; i<nextActions.getWidth() ; i++){
                in[i] = nextActions.get(0, i);
            }
            in[maxIndex] = qsa;
            agent.learn({in});

            done = fb.done;
            position = fb.position;
            eps *= epsDecay;
        }
        cout << "[" << steps << " steps]" << endl;
    }

    // testing
    cout << "\n\nPlaying game..." << endl;
    usleep(1000000);

    int action, position = game.reset();
    bool done = false;
    while(!done){
        game.render();
        std::cout << "\r";

        Matrix actions = agent.computeOutput({toCategorical(position, game.length)});
        getMax(actions, NULL, &action, NULL);
        Observation fb = game.step(action);
        done = fb.done;
        position = fb.position;

        usleep(200000);
    }
}

/** MAIN
 * 
 *   TODO
 * 
 * 
 * 
 * 
 * 
 * 
 */