#include "../include/DQN.h"

DQN::DQN(){
    learning_rate = 0.001;
    gamma = 0.99;
    eps = 1.0; // procent określający z jakim prawdopodobieństwem wykonamy ruch losowo
    epsDecay = 0.95; // procent maleje TODO
    target_agent_update_freaquency = 100;
    target_agent_count_down = target_agent_update_freaquency;
    n_steps_in_one_go = 10 * game.length();
    episode_n = 1000;
    
    
    agent = Policy(game.length(), 10,8, game.actionsCount, learning_rate);
    target_agent = agent.copy();
    network_learned = false;
    //visited = vector<bool>(game.length(),false);
    if(use_threads){
        learning_batch_size = thread::hardware_concurrency() -1;// << ilość wątków jakie będą pracowąć podczas uczenia
        updatelocalagentfrequency = 1; //! UWAGA << zmienna odpowiedzialna za częstotliwość updateowania local_agentów przy uczeniu wielowątkowym. 
    }else{
        threads_keep_working = false;
        learning_batch_size = 1;
    }
}

void DQN::resetAgents(int hidden_count,int hidden_size){
    network_learned = false;
    int hc = hidden_count;
    if(hc <= 0){
        hc = 8;
    }
    int hs = hidden_size;
    if(hs <= 0){
        hidden_size = 10;
    }
    n_steps_in_one_go = 10 * game.length();
    agent = Policy(game.length(), hs,hc, game.actionsCount, learning_rate);
    if(use_target_agent){
        target_agent = agent.copy();
    }
    memory.clear();
}

void DQN::changeGame(int sizeH,int sizeW){
    if(game.lengthH != sizeH || game.lengthW != sizeW){
        game.lengthH = sizeH;
        game.lengthW = sizeW;
    }
    game.reset();
}

bool DQN::collect_memory_step(){
    std::vector<float> oldGameRepresentation = game.getGameRepresentation();
    int action = 0;
    // take random actions sometimes to allow game exploration
    if(((float) rand() / RAND_MAX) < eps){
        action = rand()%game.actionsCount;
    }
    else{
        agent.computeOutput_fast(oldGameRepresentation).getMax( NULL, &action, NULL);
    }


    Observation fb = game.step(action);
    if(!use_memory){
        memory.clear();
    }
    else{
        if(memory.size() > max_memory_size){
            memory.erase(memory.begin(), memory.begin() +( memory.size() / 2));
        }
    }
    
    memory.push_back(DQNMemoryUnit(game.getGameRepresentation(),oldGameRepresentation,action,fb.reward,fb.done));
    return fb.done;
}



void DQN::learn_from_memory(int thread_id){
    float q_correction=0, max=0;
    DQNMemoryUnit learningExample = choose_random_from_memory();
    //cout<<"computeOutput thread:"<<thread_id<<endl;
    if(use_target_agent){
        target_agent.computeOutput(learningExample.game_next).getMax( NULL, NULL, &max);
    }else{
        agent.computeOutput(learningExample.game_next).getMax( NULL, NULL, &max);
    }
    //cout<<"q_correction thread:"<<thread_id<<endl;
    if(learningExample.done){
        q_correction = learningExample.reward;
    }else{
        q_correction = learningExample.reward + gamma*max;
    }

    agent.learn(q_correction,learningExample.action,learningExample.game);
}


void DQN::makeDQN_Thread(int thread_idx){
    cout<<"started therad "<<thread_idx<<endl;
    Environment2D local_game;
    Policy local_agent = agent.copy();
    //Policy local_target_agent = target_agent.copy();
    DQNMemoryUnit learningExample;
    float q_correction=0, max=0;

    std::unique_lock<std::mutex> start_lck(start_threaded_learning_mtx);
    start_lck.unlock();
    std::unique_lock<std::mutex> finish_lck(finished_threaded_learning_mtx);
    finish_lck.unlock();
    std::unique_lock<std::mutex> start_ulck(start_threaded_updateing_mtx);
    start_ulck.unlock();
    std::unique_lock<std::mutex> finish_ulck(finished_threaded_updateing_mtx);
    finish_ulck.unlock();

    while(threads_keep_working){
        
        start_lck.lock(); //czekamy aż Master powie że można zacząć się uczyć
        start_threaded_learning.wait(start_lck,[&]{return !thread_finished_learning[thread_idx];});//czekamy aż Master powie że można zacząć się uczyć
        start_lck.unlock();
        if(dev_debug_threading)
            cout<<"therad "<<thread_idx<<" started next learning session"<<endl;
        //start_lck.unlock();
        if(!threads_keep_working){
            return;
        }
        
        // if(target_agent_count_down == target_agent_update_freaquency)
        //     local_target_agent.updateParameters(target_agent);//! //TODO //!
        if(dev_debug_threading)
            cout<<"therad "<<thread_idx<<" is learning..."<<endl;

        learningExample = choose_random_from_memory();

        if(dev_debug_threading)
            cout<<"therad "<<thread_idx<<" choose_random_from_memory"<<endl;

        if(learningExample.done){
            q_correction = learningExample.reward;
        }else{
            target_agent.computeOutput_fast(learningExample.game_next).getMax( NULL, NULL, &max);//! //TODO //!
            q_correction = learningExample.reward + gamma*max;
        } 
        
        if(dev_debug_threading)
            cout<<"therad "<<thread_idx<<"computed output"<<endl;

        local_agent.learn(q_correction,learningExample.action,learningExample.game,false,1.0);// ! tutaj można dodać liczbę o jaką będą przeskalowane zmiany z uczenia (ostatnia wartość)

        if(dev_debug_threading)
            cout<<"therad "<<thread_idx<<"local_agent.learn"<<endl;
        
        if(dev_debug_threading)
            cout<<"therad "<<thread_idx<<"change_weights_by_other_policy PRE"<<endl;
        change_weigths_of_global_agent.lock();

        agent.change_weights_by_other_policy(&local_agent);
        
        change_weigths_of_global_agent.unlock();

        if(dev_debug_threading)
            cout<<"therad "<<thread_idx<<"change_weights_by_other_policy POST"<<endl;

        //informujemy że skończyliśmy uczyć
        if(dev_debug_threading)
            cout<<"therad "<<thread_idx<<" finished learning..."<<endl;

        thread_finished_learning[thread_idx] = true; // zaznaczamy że skończyliśmy uczyć
        finish_lck.lock();
        finished_threaded_learning.notify_one();
        finish_lck.unlock();
        
        local_agent.change_weights(true);

        if(local_game.check_if_good_enougth(&local_agent)){
            final_agent = local_agent;
            network_learned = true;
            cout<<"therad "<<thread_idx<<" HAVE A SOLUTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<endl;
            //showBestChoicesFor(local_agent);
        }
        
        
        //czekamy aż Master powie że można zupdateować local agenta
        if(dev_debug_threading)
            cout<<"therad "<<thread_idx<<" whaitibng for weigths update approval"<<endl;
        start_ulck.lock();
        start_threaded_updateing.wait(start_ulck,[&]{
            return !thread_finished_learning[thread_idx];
        });
        start_ulck.unlock();

        
        //lck.unlock();true== 0){
        local_agent.updateParameters(&agent);
        if(dev_debug_threading)
            cout<<"therad "<<thread_idx<<" updateing local_agent"<<endl;
        

        
        thread_finished_learning[thread_idx] = true; // zaznaczamy że skończyliśmy udate
        finish_ulck.lock();
        finished_threaded_updateing.notify_all();
        finish_ulck.unlock();
        
        if(dev_debug_threading)
            cout<<"therad "<<thread_idx<<" whaitibng for next learning session to start"<<endl;
        //czekamy aż Master powie że można zacząć się uczyć dalej
        }
}

Policy DQN::train(double* learning_time,int* steps_done,int* episodes){
    assert(learning_batch_size <= threads_numer);
    bool whait_for_update = false;
    int done_counter = 0;
    if(show_output)
        cout << "Start training,  episodes:"<<episode_n<<endl;
    auto start_time = chrono::steady_clock::now();
    int i;
    
    int steps=0;
    bool done=false;
    std::unique_lock<std::mutex> s_lck(start_threaded_learning_mtx);
    s_lck.unlock();
    std::unique_lock<std::mutex> f_lck(finished_threaded_learning_mtx);
    f_lck.unlock();
    std::unique_lock<std::mutex> start_lck(start_threaded_updateing_mtx);
    start_lck.unlock();
    std::unique_lock<std::mutex> finish_lck(finished_threaded_updateing_mtx);
    finish_lck.unlock();
    if(use_threads)
        for(int thread_id = 0; thread_id < learning_batch_size; thread_id++){
            thread_finished_learning[thread_id] = true;
            threads[thread_id] = std::thread([this,thread_id](){this->makeDQN_Thread(thread_id);});
        }

    for (i=0 ; i<episode_n ; i++){
        steps=0;
        done=false;
        
        game.reset();
        while(!done && steps<n_steps_in_one_go && network_learned == false){
            //cout<<"start sampling"<<endl
            if(dev_debug_threading)
                cout<<"MAIN_TRAIN collecting memory steps, steps:"<<steps<<endl;
            for(int b = 0; b<learning_batch_size && done == false; b++){
                steps++;
                done = collect_memory_step();
            }
            //cout<<"start learning"<<endl;
            if(use_threads){
                // if(whait_for_update){
                //     if(dev_debug_threading)
                //         cout<<"MAIN_TRAIN w8 for update to come to the end"<<endl;
                // }else{
                //     whait_for_update = true;
                // }

                
                if(dev_debug_threading)
                    cout<<"MAIN_TRAIN inform that threads can start lerning"<<endl;
                s_lck.lock();// ustawiamy flagi uczenia 
                for(int b = 0; b<learning_batch_size  && b<memory.size(); b++){
                    thread_finished_learning[b] = false;
                    if(dev_debug_threading)
                        cout<<"MAIN_TRAIN inform that thread:"<<b<<" can start lerning:"<<!thread_finished_learning[b]<<endl;
                }
                start_threaded_learning.notify_all();// powiadamiamy wszystkie wątki że można uczyć
                s_lck.unlock();
                
                
                if(dev_debug_threading)
                    cout<<"MAIN_TRAIN started whaiting for threaded learning to end"<<endl;
                f_lck.lock();
                finished_threaded_learning.wait(f_lck,[&](){
                    for(int b = 0; b<learning_batch_size  && b<memory.size(); b++){
                        if(thread_finished_learning[b] == false){
                            return false;
                        }
                    }
                    return true;
                });
                f_lck.unlock();
                
                // powiadamiamy wszystkie wątki że można zupdateować swojego local agenta
                if(dev_debug_threading)
                    cout<<"MAIN_TRAIN let threads update"<<endl;
                start_lck.lock();
                for(int b = 0; b<learning_batch_size  && b<memory.size(); b++){
                    thread_finished_learning[b] = false;// ustawiamy flagi updateu 
                }
                start_threaded_updateing.notify_all();// powiadamiamy wszystkie wątki że można zupdateować swojego local agenta
                start_lck.unlock();
                


                if(game.check_if_good_enougth(&agent)){
                    network_learned = true;
                    final_agent = agent;
                    cout<<"MAIN HAVE A SOLUTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<endl;
                    //showBestChoicesFor(local_agent);
                }
                // if(dev_debug_threading)
                //     cout<<"MAIN_TRAIN change_weights"<<endl;

                // agent.change_weights();
                // czekamy na koniec updateowania agenta
                if(dev_debug_threading)
                    cout<<"MAIN_TRAIN let threads update"<<endl;
                finish_lck.lock();// czekamy na koniec updateowania agenta
                finished_threaded_updateing.wait(finish_lck,[&](){
                    for(int b = 0; b<learning_batch_size  && b<memory.size(); b++){
                        if(thread_finished_learning[b] == false){
                            return false;
                        }
                    }
                    return true;
                });
                finish_lck.unlock();
                
            }else{
                learn_from_memory(-1);    
            }
            if(use_target_agent)
                if(target_agent_count_down <= 0){
                    target_agent.updateParameters(&agent);
                    target_agent_count_down = target_agent_update_freaquency;
                }else{
                    //network_learned = game.check_if_good_enougth(agent);
                    target_agent_count_down -= learning_batch_size;
                    eps *= epsDecay;
                }
            if(done|| steps>=n_steps_in_one_go){
                done_counter += steps;
                //network_learned = game.check_if_good_enougth(agent);
                if(use_target_agent){
                    target_agent.updateParameters(&agent);
                    target_agent_count_down = target_agent_update_freaquency;
                }
            }


            if((steps >= n_steps_in_one_go && eps < 0.01) || (eps < 0.00001)){ // reseting exploration chance
                eps = 1.0;
            }
        }

        if(show_output){
            cout << "Episode " << i+1 << "/" << episode_n << "\t";
            cout << "[" << steps << " steps] eps:"<< eps <<"        total_steps_done:"<<done_counter <<endl ;//<<endl << " Szansa ma losowy krok" << eps*100.0<<endl;
            if(network_learned){
                cout<<"Showing best choises produced by the winner"<<endl;
                showBestChoicesFor(final_agent);
            }else{
                showBestChoicesFor(agent);
            }
        }

        if(network_learned){
            threads_keep_working = false;
            std::unique_lock<std::mutex> start_lck(start_threaded_learning_mtx);// ustawiamy flagi uczenia 
            for(int b = 0; b<learning_batch_size  && b<memory.size(); b++){
                thread_finished_learning[b] = false;
            }
            //cout<<"MAIN_TRAIN starting learning"<<endl;
            start_threaded_learning.notify_all();// powiadamiamy wszystkie wątki że można uczyć
            start_lck.unlock();
            for(int b = 0; b<learning_batch_size  && b<memory.size(); b++){
                        if (threads[b].joinable())
                            threads[b].join();
                    }
            break;
        }
    }
    exec_time = chrono::steady_clock::now() - start_time;
    //odp1 = agent.computeOutput({game.getGameRepresentation()});
    //odp2 = target_agent.computeOutput({game.getGameRepresentation()});
    //cout <<"agent        "<< odp1 <<"target_agent "<< odp2 << endl;
    if(show_output)
        cout<<" Training took:"<<exec_time.count() / 1000.0<<"s"<<endl;
    if(learning_time != NULL)
        *learning_time = exec_time.count() / 1000.0;
    if(steps_done != NULL)
        *steps_done = done_counter;
    if(episodes != NULL)
        *episodes = i;
    return final_agent;
}

void DQN::showBestChoicesFor(Policy agent){ 
    int action,w = 0;
    float max = 0.0;
    for(int h = 0;h<game.lengthH;h++){
        for(w = 0;w<game.lengthW;w++){
            agent.computeOutput_fast(game.toGameRepresentation(h,w)).getMax( NULL, &action, &max);
            if(action==0){
                if(isnan(max))
                    std::cout<<"!";
                std::cout<<"U";
            }else if(action==1){
                std::cout<<"D";
            }else if(action==2){
                std::cout<<"L";
            }else if(action==3){
                std::cout<<"R";
            }
        }
        std::cout<<std::endl;
    }
}

DQNMemoryUnit DQN::choose_random_from_memory(){ 
    //if(use_memory && give_last < 0)
        //cout<<"getting random from memory, size "<<memory.size()<<endl;
        return memory[(int)(rand() % memory.size())];
    //return memory[memory.size()-1-give_last];
}

//TODO

// nie ma co updateować agenta w wątkach trzeba jedynie przekazać pochodne po W/B i main to sobie zupdateuje. ???

// main nie powinien czekać aż wątki skończą updateować, przystepujemy od razu do zbieranie stepsów ? 

// 