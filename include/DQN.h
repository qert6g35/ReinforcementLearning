#ifndef dqn
#define dqn

#include <stdlib.h>
#include <time.h>

#include "../include/matrix.h"
#include "../include/environment.h"
#include "../include/policy.h"

using namespace std;

struct DQN{
    Environment game; // init environment
    Policy agent = Policy(game.length, 10,2, game.actionsCount, 0.2);
    Policy target_agent = agent.copy();
    

    double gamma = 0.8;
    double eps = 1.0; // procent określający z jakim prawdopodobieństwem wykonamy ruch losowo
    double epsDecay = 0.995; // procent maleje TODO

    int episode_n = 100;

    Matrix odp1 = agent.computeOutput({game.getGameRepresentation()});
    Matrix odp2 = target_agent.computeOutput({game.getGameRepresentation()});
    
    Policy train(){

        cout << odp1 << endl << odp2 << endl;
        cout << "Start training,  episodes:"<<episode_n<<endl;
        for (int i=0 ; i<episode_n ; i++){
            int steps=0, maxIndex=0, action=0;
            double qsa=0, max=0;
            bool done=false;
            std::vector<double> correction;
            Matrix Qaprox;
            
            game.reset();

            int stepper = 0;
            while(!done){
                stepper++;
                game.render();
                std::cout << "\r";

                Matrix actions = agent.computeOutput({game.getGameRepresentation()});
                actions.getMax( NULL, &action, NULL);
                Observation fb = game.step(action);
                done = fb.done;
                //position = fb.position;
                if(stepper > 30){
                    break;
                }
                usleep(25000);
            }

            game.reset();
            done=false;
            while(!done && steps<300){
                steps++;
                // save enviroment before takeing action
                std::vector<double> oldGameRepresentation = game.getGameRepresentation();

                //aproximateing q table
                Qaprox = agent.computeOutput({game.getGameRepresentation()});

                // take random actions sometimes to allow game exploration
                if(((double) rand() / RAND_MAX) < eps){
                    action = rand()%game.actionsCount;
                }
                else{
                    //picking action that follows Q-table
                    Qaprox.getMax( NULL, &action, NULL);
                }

                // take action in enviroment
                //Qaprox.print(cout);
                Observation fb = game.step(action);
                done = fb.done;
                //
                //
                //
                // get best action in next state
                Matrix Qprox_next = agent.computeOutput({game.getGameRepresentation()});//TODO tutaj sieć ma już inne wyjście policzone (ni to dla którego jest obecna nagroda!!!!)
                //Qprox_next.print(cout);
                // maxIndex <to> akcja która jako następna wdłg naszego oszacowania jest najleprsza (najlepsza następna akcja)
                // max <to> oszacowana wartosć Q tej najlepszej akcji
                Qprox_next.getMax( NULL, NULL, &max);
                // parametr QSA <to> R_s + wsp * wartość następnej najlepszej akcji
                if(done == true){
                    qsa = fb.reward;
                }else{
                    qsa = fb.reward + gamma*max;
                }
                //Qaprox.print(cout);
                correction = Qaprox.getRow(0);
                correction[action] = qsa;

                agent.learn(correction,oldGameRepresentation);

                eps *= epsDecay;
            }
            if(i%10 == 0 || i%10 == 1){
                cout << "Episode " << i+1 << "/" << episode_n << "\t";
                cout << "[" << steps << " steps] eps:"<< eps << endl ;//<<endl << " Szansa ma losowy krok" << eps*100.0<<endl;
            }
            if(steps == 300 && eps < 0.001){
                eps = 1.0;
            }

        }

        odp1 = agent.computeOutput({game.getGameRepresentation()});
        odp2 = target_agent.computeOutput({game.getGameRepresentation()});
        cout << odp1 << endl << odp2 << endl;
        return agent;
    }
};


#endif