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
    Policy agent = Policy(game.length, 10, game.actionsCount, 0.1);

    double gamma = 0.8;
    double eps = 1.0; // procent określający z jakim prawdopodobieństwem wykonamy ruch losowo
    double epsDecay = 0.95; // procent maleje TODO

    //int episode = 100;

    Policy train(int episode = 100){
        cout << "Start training,  episodes:"<<episode<<endl;
        for (int i=0 ; i<episode ; i++){
            //if(i%10 == 9){
                cout << "Episode " << i+1 << "/" << episode << "\t";
            //}
            int steps=0, maxIndex=0, action=0;
            double qsa=0, max=0;
            bool done=false;
            
            game.reset();

            while(!done && steps<300){
                steps++;

                // take random actions sometimes to allow game exploration
                if(((double) rand() / RAND_MAX) < eps){
                    action = rand()%game.actionsCount;
                }
                else{
                    //aproximateing q table
                    Matrix Qaprox = agent.computeOutput({game.getGameRepresentation()});
                    //picking action that follows Q-table
                    Qaprox.getMax( NULL, &action, NULL);
                }

                // take action
                Observation fb = game.step(action);
                //
                //
                //
                // get max action in next state
                Matrix Qprox_next = agent.computeOutput({game.getGameRepresentation()});//TODO tutaj sieć ma już inne wyjście policzone (ni to dla którego jest obecna nagroda!!!!)
                Qprox_next.print(cout);
                // maxIndex <to> akcja która jako następna wdłg naszego oszacowania jest najleprsza (najlepsza następna akcja)
                // max <to> oszacowana wartosć Q tej najlepszej akcji
                Qprox_next.getMax( NULL, &maxIndex, &max);
                // parametr QSA <to> R_s + wsp * wartość następnej najlepszej akcji
                qsa = fb.reward + gamma*max;

                // tworzymy wektór o rozmiarze wyjścia sieci 
                std::vector<double> in(game.actionsCount);
                // kopiujemy Aproksymację następnych ruchów //! Matrix > Vector !! 
                for(int i=0 ; i<Qprox_next.getWidth() ; i++){
                    //in[i] = Qprox_next.get(0, i);
                    //? used below to check in network is working corectly
                    in[i] =i;
                }
                // podmiana wartości oszacowanej ???? //! przez to co tu się dzieje ten program w zaszdzie może mocno nie działać !!
                //in[maxIndex] = qsa;
                agent.learn({in});

                done = fb.done;
                eps *= epsDecay;
            }
            //if(i%10 == 9){
                cout << "[" << steps << " steps]" << endl;
            //}
        }

        return agent;
    }
};


#endif