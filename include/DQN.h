#ifndef dqn
#define dqn

#include <stdlib.h>
#include <time.h>
#include <random>

#include "../include/matrix.h"
#include "../include/environment.h"
#include "../include/policy.h"

using namespace std;
const bool show_output = false;

struct DQNMemoryUnit
{
    std::vector<double> game = std::vector<double>();
    std::vector<double> game_next = std::vector<double>();
    int action;
    double reward; 
    bool done;

    DQNMemoryUnit(std::vector<double> ngame,std::vector<double> ngame_next,int naction,double nreward,bool isDone){
        game = ngame;
        game_next = ngame_next;
        action = naction;
        reward = nreward;
    }
};


DQNMemoryUnit choose_random_from_(std::vector<DQNMemoryUnit> memory,mt19937 gen){
    uniform_int_distribution<int> distrib(0, memory.size()-1);
    int example = distrib(gen);//memory.size() - 1;//
    //cout<<" e:"<<example<<" mem_size:"<<memory.size()<<endl;
    return memory[example];
}


struct DQN{
    Environment1D game; // init environment

    Policy agent = Policy(game.length, 10,4, game.actionsCount, 0.05);
    Policy target_agent = agent.copy();

    std::vector<DQNMemoryUnit> memory;

    double gamma = 0.8;
    double eps = 1.0; // procent określający z jakim prawdopodobieństwem wykonamy ruch losowo
    double epsDecay = 0.995; // procent maleje TODO
    int target_agent_update_freaquency = 5;
    int target_agent_count_down = target_agent_update_freaquency;
    int n_steps_in_one_go = 300;
    random_device rd;
    mt19937 gen = mt19937(rd());

    int episode_n = 100;

    Matrix odp1 = agent.computeOutput({game.getGameRepresentation()});
    Matrix odp2 = target_agent.computeOutput({game.getGameRepresentation()});

    Policy train(){
        int done_counter = 0;
        //cout <<"agent        "<< odp1 <<"target_agent "<< odp2 << endl;
        cout << "Start training,  episodes:"<<episode_n<<endl;
        for (int i=0 ; i<episode_n ; i++){
            int steps=0, maxIndex=0, action=0;
            double q_correction=0, max=0;
            bool done=false;
            Matrix Qaprox;
            
            game.reset();

            int stepper = 0;

            if(odp1.haveAnyNan() && show_output)
                cout <<"agent output:"<< odp1 <<"\n";
            done=false;
            while(!done && steps<n_steps_in_one_go){
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
                memory.push_back(DQNMemoryUnit(game.getGameRepresentation(),oldGameRepresentation,action,fb.reward,done));
                //DQNMemoryUnit learningEgxample = memory[memory.size() - 1];//choose_random_from_(memory,gen);
                DQNMemoryUnit learningEgxample = choose_random_from_(memory,gen);
                // get best action in next state
                Matrix Qprox_next = target_agent.computeOutput(learningEgxample.game_next);//TODO tutaj sieć ma już inne wyjście policzone (ni to dla którego jest obecna nagroda!!!!)
                //Qprox_next.print(cout);
                // maxIndex <to> akcja która jako następna wdłg naszego oszacowania jest najleprsza (najlepsza następna akcja)
                // max <to> oszacowana wartosć Q tej najlepszej akcji
                Qprox_next.getMax( NULL, NULL, &max);
                // parametr QSA <to> R_s + wsp * wartość następnej najlepszej akcji
                if(learningEgxample.done == true){
                    q_correction = learningEgxample.reward;
                }else{
                    q_correction = learningEgxample.reward + gamma*max;
                }
                //Qaprox.print(cout);
                //cout<<"corection "<<q_correction<<" on action:"<< learningEgxample.action<<endl;
                agent.learn(q_correction,learningEgxample.action,learningEgxample.game);
                //agent.learn(-1,0,learningEgxample.game);

                eps *= epsDecay;
                if(target_agent_count_down == 0){
                    if(agent.computeOutput(game.toGameRepresentation(0,game.length)).haveAnyNan()){
                        agent.updateParameters(target_agent);
                    }else{
                        //cout<<"update target_agent";
                        target_agent.updateParameters(agent);
                    }
                    target_agent_count_down = target_agent_update_freaquency;
                }else{
                    target_agent_count_down--;
                }
            }
            target_agent.updateParameters(agent);
            target_agent_count_down = target_agent_update_freaquency;

            if(show_output){
                cout << "Episode " << i+1 << "/" << episode_n << "\t";
                cout << "[" << steps << " steps] eps:"<< eps << endl ;//<<endl << " Szansa ma losowy krok" << eps*100.0<<endl;
            }
            if(steps == n_steps_in_one_go && eps < 0.0001){
                eps = 1.0;
            }
            if(steps == n_steps_in_one_go && show_output){
                odp1 = agent.computeOutput({game.getGameRepresentation()});
                cout <<"agent        "<< odp1<< endl;
            }
            steps = 0;
            done = false;
            game.reset();
            while(!done){
                stepper++;
                if(show_output){
                    game.render();
                    std::cout << "\r";
                }
                Matrix actions = target_agent.computeOutput({game.getGameRepresentation()});
                actions.getMax( NULL, &action, NULL);
                Observation fb = game.step(action);
                done = fb.done;
                //position = fb.position;
                if(stepper > game.length+1){
                    break;
                }
                if(show_output)
                    usleep(25000);
            }

            if(done == true){
                done_counter += 1;
            }else{
                done_counter = 0;
            }
            if(done_counter == 3){
                break;
            }

        }

        //odp1 = agent.computeOutput({game.getGameRepresentation()});
        //odp2 = target_agent.computeOutput({game.getGameRepresentation()});
        //cout <<"agent        "<< odp1 <<"target_agent "<< odp2 << endl;
        return target_agent;
    }
};


#endif
/*







eksplodujący gradient, wartości zaczynają lecieć w nieskończoność + lub - 
agent -2.34e+34 -9.606e+31 i leci const correction -81 action 0 
agent  -3  i - 40                         
agent -0.008 i -0.0917 corection const = -1 na action 0

tutaj zawsze idziemy w lewo, mimo że epsilon jest srestartowany, na każdeym resecie jest szansa że wartości zwaracane przez agenta będą jeszcze gorsze
w momencie resetu (kiedy zablokowany alg zreseruje wsp. eksploracji to na zaminę pogarsza ) 



po zdjęciu blokady na MAX ujemną wartość (-100 było najmniejszą możliwą wartością w getMax)

agent się poddał i wyrzuca -nan -nan













*/