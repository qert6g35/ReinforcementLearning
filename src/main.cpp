#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <cmath>

#include "../include/environment.h"
#include "../include/DQN.h"

using namespace std;

void shuffle(int arr[], int n) {
    for (int i = n - 1; i >= 1; i--) {
        int j = rand() % (i + 1);
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}

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

void run_multithreaded_tests(){
    int n_samples = 1;
    //Environment2D game = Environment2D(); 

    std::ofstream writeHere;
    //writeHere.open("work_balance_data/"+std::to_string(test_num)+"/DQN_multithreaded.csv", std::ios::app);
    writeHere.open("work_balance_data/DQN_multithreaded.csv", std::ios::app);
    
    DQN trainer;
    
    double time = 0;
    double prev_time = 2;
    int steps = 0;
    int threds_number[11] = {128,64,1,2,4,6,8,12,16,24,32};
    shuffle(threds_number,11);
    // cout<<"start sampling in order:"<<endl;
    // for(int threads_number_id = 0; threads_number_id < 12; threads_number_id++){
    //     cout<<threds_number[threads_number_id]<<" threds"<<endl;
    // }
    // sleep(5);
    //return;
    
    for(int helper = 0; helper<n_samples;helper++){
        trainer.resetAgents(10,10);
        Policy agent_prime = trainer.get_agent().copy();
        cout<<"\n start loop "<<helper<<" \n"<<endl;
        for(int threads_number_id = 0; threads_number_id < 11; threads_number_id++){
            cout<<"start sampling for "<<threds_number[threads_number_id]<<" threds"<<endl;
            trainer.resetAgents(10,10,threds_number[threads_number_id]);
            trainer.folder_to_safe_to = helper;
            trainer.set_agent(agent_prime.copy());
            trainer.train(&time,&steps,NULL);
            //game.show_how_it_works(agent);
            writeHere<<helper<<","<<threds_number[threads_number_id]<<","<<time<<","<<steps<<endl;
            if((time < 0 && prev_time < 0) || (time < 0 && steps >= 1000000) ){
                cout<<"LEARNING JAMMED"<<endl;
                return;
            }
        }
        
    }
    writeHere.close();
}

void run_multithreaded_tests_for_number_of_env_stepping(){
    int n_samples = 5;
    //Environment2D game = Environment2D(); 

    std::ofstream writeHere;
    writeHere.open("DQN_multithreaded_COMPARE_ENV_STEPPING_METHOOD.csv", std::ios::app);
    
    DQN trainer;
    double time = 0;
    int steps = 0;
    for(int threads_number_id = 0; threads_number_id < 10; threads_number_id++){
        //cout<<"start sampling for "<<threds_number[threads_number_id]<<" threds"<<endl;
        for(int helper = 0; helper<n_samples;helper++){
            trainer.make_only_one_learning_steps_ALWAYS = false;
            trainer.resetAgents(10,10);
            Policy agent = trainer.train(&time,&steps,NULL);
            //game.show_how_it_works(agent);
            writeHere<<"0,"<<time<<","<<steps<<endl;

            trainer.make_only_one_learning_steps_ALWAYS = true;
            trainer.resetAgents(10,10);
            agent = trainer.train(&time,&steps,NULL);
            //game.show_how_it_works(agent);
            writeHere<<"1,"<<time<<","<<steps<<endl;
        }
        
    }
    writeHere.close();
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
    
    // trainer.resetAgents();

    // Policy agent2 = trainer.train(&time,&steps,&episodes);//uczymy nowego agenta

    // cout << "trained for:" << time << "s" << " thats "<<episodes<<" episodes done in "<<steps<<" steps"<<endl;

    // cout << "\n\nPlaying game..." << endl;
    // usleep(100000);

    // game.show_how_it_works(agent);
}

int main(int argc, char *argv[]){
    srand (time(NULL)); // to generate random weights
    //cout<<"we will be running "<<thread::hardware_concurrency()<<" threads"<<endl;
    // finish_time_tests(8);
    // run_time_tests(9,8);
    //testSingleExaple();
    //show_how_program_works();
    // for(int i = 1; i < 1000; i ++){
    //     cout<<" STARTING TEST-SET NO."<<i<<endl;
    run_multithreaded_tests();
    // }
    return 0;
}

/** MAIN
 * //! zastanów się, czy nie chcemy użyć np TensorFlow czy innej biblioteki do zastąpienia Polityki
 *   


 *TODO w drobno: (komunikacja wystepuje bardzo często między wątkami)
 *TODO 
        <jak chcemy to zrealizować?>
        ** wątki spawnowane mają za zadanie wykonać funkcję learn_from_memory(); (odpowiednio blokując się na wzajem na mutexie podczas updateowania wag.)
    
 *TODO 

0) zmnijszamy ilosć max iteracji względem ilości zespawnowanych wątków // chyba nie trzeba

1) raz spawnujemy thready i tylko komunikacja //* DONE 

__ od tąd już chyba zmiany przechodzą w gruboziarnistość___

2) kżdy thread liczy na niezależnym agencie + synchronizacja z głównym wątkiem

3) przed updatem każdy wątek sprawdza czy same jego zmiany wystarczą do tego aby sieć już działała poprawnie













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
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 *? Nowa ocena kodu i wyplutych wyników:
 * 
 * brakuje czasów działania wątków. Musimy zbierać czasy (timestampy) pracy każdego wątku. Czyli 
 * timestampy:
 ** uczenia
 ** updateowania local agentów
 ** zbierania traceów
 ** czekania aż local_agenty się zupdateują
 * 
 * do tego analiza czasu wypałnienia pracy, czyli chcemy wiedzieć ile czasu wątki uczą agentów, 
 * a ile czasu czekamy na główny wątek ..tylko..po..co.. 
 * 
 * ? z prawa amdala możemy wyliczyć : S_ub (górna granica przyśpieszenia)
 * 
 * ? obliczenia pozwalają nam wyznaczyć P
 * * P - część programu którą da się zrównoleglić 
 * 
 * P można też wyestymować z experymentalnego przyśpieszenia
 * 
 * ? można te dwa czasy ze sobą porównać 
 * 
 * ? c ojeszcze można dzięki tym badaniom zobaczyc? 
 * 
 */