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
    
    DQN trainer;//= DQN();
    double time = 0;
    int steps = 0;
    for(int iH = 9; iH <= instance_max_size; iH++){
        for(int iW = 2; iW <= iH; iW++){
            trainer.changeGame(iH,iW);
            cout<<"start sampling for heigth:"<<iH<<", width:"<<iW<<endl;
            for(int s=0;s<n_samples;s++){
                    trainer.resetAgents();
                    trainer.train(&time,&steps,NULL);
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
    
    DQN trainer;
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
                    trainer.train(&time,&steps,NULL);
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

    DQN trainer;
    double time = 0;
    int steps = 0;
    int episodes = 0;
    Environment2D game = Environment2D(); // init environment
    // game.render();
    // game.step(3);
    // game.step(1);
    // usleep(1000000);
    // game.render();
    Policy agent = trainer.train(&time,&steps,&episodes);//uczymy nowego agenta

    cout << "trained for:" << time << "s" << " thats "<<episodes<<" episodes done in "<<steps<<" steps"<<endl;

    cout << "\n\nPlaying game..." << endl;
    usleep(100000);

    game.show_how_it_works(agent);
}

int main(int argc, char *argv[]){
    //srand (time(NULL)); // to generate random weights

    // finish_time_tests(8);
    // run_time_tests(9,8);
    //testSingleExaple();
    show_how_program_works();
    
    return 0;
}

/** MAIN
 * //! zastanów się, czy nie chcemy użyć np TensorFlow czy innej biblioteki do zastąpienia Polityki
 *   
 *   drobno vs gruboziarnistość
 *TODO w drobno: (komunikacja wystepuje bardzo często między wątkami)
 *TODO 
        <jak chcemy to zrealizować?>
        ** wątki spawnowane mają za zadanie wykonać funkcję learn_from_memory(); (odpowiednio blokując się na wzajem na mutexie podczas updateowania wag.)
    
 *TODO 
 *TODO w grubo: (komunikacja występuje rzadziej)
 *TODO      
      wątki - (tutaj chcemy jak najbardziej skożystać ze wspólnej pamięci) 
    ** w0 Parameter Server 
    ** wątki zbierające Tracey i uczące, każdy osobnego agenta
    ** memory przechowywane w wariancie jednej dużej pamięci
    *! ważne aby uczenie i zbieranie traceów ze sobą nie kolidowały, my chcemy i tak zoptymalizować uczenie bo to zjmuje najwięcej czasu

    ** każdy z podwątków wysyła zmiany do w0 który updateuje Q-server na bierząco
    ** wątki updateują swojego agenta względem Q-servera i target-agenta raz na jakiś czas względem Q-servera (totaj pole do popisu dla zmian)

 *TODO          
      procesy - każdy z osobna wykonuje funkcę train(), 
*TODO komunikacja odbywa się jedynie gdy któryś z nich znajdzie rozwiązanie, wtedy brodeactujemy do reszty że robota wykonana

    ** tutaj można zrobić tak jak na wątkach z tym wyjątkiem że każdy ma swoje memory więc jedyne co synchronizujemy to sieć w jedną i drógą stronę


 OGÓLNIE DLA GRUBO :
    ** tworzy się zbiory po 2 agentów, jeden generuje dane, drógi przeprowadza proces uczenia, oba updateują swoje dane (u nas lepiej mieć 1/2 agenta generującego Tracey, i więcej agentów do uczenia) 




 *TODO      wątki vs procesy   
 *TODO        1) rozproszoność (procesy) vs 2) równoległość (threads) w ramach efektywności
 *TODO  
 *TODO      sprawdzić czas generacji nowych sieci
 *TODO 
 *TODO 
 * 
 * 
 */