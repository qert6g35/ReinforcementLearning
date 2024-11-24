#ifndef dqn
#define dqn

#include <stdlib.h>
#include <time.h>
#include <random>

#include "../include/matrix.h"
#include "../include/environment.h"
#include "../include/policy.h"

using namespace std;

struct DQNMemoryUnit
{
    std::vector<double> game = std::vector<double>();
    std::vector<double> game_next = std::vector<double>();
    int action;
    double reward; 

    DQNMemoryUnit(std::vector<double> ngame,std::vector<double> ngame_next,int naction,double nreward){
        game = ngame;
        game_next = ngame_next;
        action = naction;
        reward = nreward;
    }
};


DQNMemoryUnit choose_random_from_(std::vector<DQNMemoryUnit> memory,mt19937 gen){
    uniform_int_distribution<int> distrib(0, memory.size()-1);
    int example = distrib(gen);//memory.size() - 1;
    //cout<<" e:"<<example<<" mem_size:"<<memory.size()<<endl;
    return memory[example];
}


struct DQN{
    Environment game; // init environment

    Policy agent = Policy(game.length, 10,2, game.actionsCount, 0.2);
    Policy target_agent = agent.copy();

    std::vector<DQNMemoryUnit> memory;

    double gamma = 0.8;
    double eps = 1.0; // procent określający z jakim prawdopodobieństwem wykonamy ruch losowo
    double epsDecay = 0.995; // procent maleje TODO
    int target_agent_update_freaquency = 50;
    int target_agent_count_down = target_agent_update_freaquency;
    random_device rd;
    mt19937 gen = mt19937(rd());

    int episode_n = 100;

    Matrix odp1 = agent.computeOutput({game.getGameRepresentation()});
    Matrix odp2 = target_agent.computeOutput({game.getGameRepresentation()});
    
    Policy train(){
        //cout << odp1 << endl << odp2 << endl;
        cout << "Start training,  episodes:"<<episode_n<<endl;
        for (int i=0 ; i<episode_n ; i++){
            int steps=0, maxIndex=0, action=0;
            double q_correction=0, max=0;
            bool done=false;
            Matrix Qaprox;
            
            game.reset();

            // int stepper = 0;
            // while(!done){
            //     stepper++;
            //     game.render();
            //     std::cout << "\r";

            //     Matrix actions = agent.computeOutput({game.getGameRepresentation()});
            //     actions.getMax( NULL, &action, NULL);
            //     Observation fb = game.step(action);
            //     done = fb.done;
            //     //position = fb.position;
            //     if(stepper > 30){
            //         break;
            //     }
            //     usleep(25000);
            // }

            game.reset();
            done=false;
            while(!done && steps<300){
                steps++;
                // save enviroment before takeing action
                std::vector<double> oldGameRepresentation = game.getGameRepresentation();

                // take random actions sometimes to allow game exploration
                if(((double) rand() / RAND_MAX) < eps){
                    action = rand()%game.actionsCount;
                }
                else{
                    //picking action that follows Q-table
                    agent.computeOutput({game.getGameRepresentation()}).getMax( NULL, &action, NULL);
                }

                // take action in enviroment
                //Qaprox.print(cout);
                Observation fb = game.step(action);
                done = fb.done;
                //
                // saveing that moment in 
                memory.push_back(DQNMemoryUnit(game.getGameRepresentation(),oldGameRepresentation,action,fb.reward));
                //DQNMemoryUnit learningEgxample = memory[memory.size() - 1];//choose_random_from_(memory,gen);
                DQNMemoryUnit learningEgxample = choose_random_from_(memory,gen);
                // get best action in next state
                Matrix Qprox_next = target_agent.computeOutput(learningEgxample.game_next);//TODO tutaj sieć ma już inne wyjście policzone (ni to dla którego jest obecna nagroda!!!!)
                //Qprox_next.print(cout);
                // maxIndex <to> akcja która jako następna wdłg naszego oszacowania jest najleprsza (najlepsza następna akcja)
                // max <to> oszacowana wartosć Q tej najlepszej akcji
                Qprox_next.getMax( NULL, NULL, &max);
                // parametr QSA <to> R_s + wsp * wartość następnej najlepszej akcji
                if(done == true){
                    q_correction = learningEgxample.reward;
                }else{
                    q_correction = learningEgxample.reward + gamma*max;
                }
                //Qaprox.print(cout);

                agent.learn(q_correction,learningEgxample.action,learningEgxample.game);

                eps *= epsDecay;
                if(target_agent_count_down == 0){
                    target_agent.updateParameters(agent);
                    target_agent_count_down = target_agent_update_freaquency;
                }else{
                    target_agent_count_down--;
                }
            }
            //if(i%10 == 0 || i%10 == 1){
                cout << "Episode " << i+1 << "/" << episode_n << "\t";
                cout << "[" << steps << " steps] eps:"<< eps << endl ;//<<endl << " Szansa ma losowy krok" << eps*100.0<<endl;
            //}
            if(steps == 300 && eps < 0.0001){
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