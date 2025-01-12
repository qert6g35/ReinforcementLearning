#include "../include/DQN.h"

DQN::DQN(){
    learning_rate = 0.001;
    gamma = 0.99;
    eps = 1.0; // procent określający z jakim prawdopodobieństwem wykonamy ruch losowo
    epsDecay = 0.98; // procent maleje TODO
    target_agent_update_freaquency = 100;
    target_agent_count_down = target_agent_update_freaquency;
    n_steps_in_one_go = 10 * game.length();
    
    
    agent = Policy(game.length(), 10,10, game.actionsCount, learning_rate);
    target_agent = agent.copy();
    network_learned = false;
    //visited = vector<bool>(game.length(),false);
    if(use_threads){
        learning_batch_size = thread::hardware_concurrency();// << ilość wątków jakie będą pracowąć podczas uczenia
        update_local_agent_frequency = 10; //! UWAGA << zmienna odpowiedzialna za częstotliwość updateowania local_agentów przy uczeniu wielowątkowym. (ile mamy czekać między updateami thread_agentow, względem głównego)
        for(int i =0 ; i < 129; i++){
            learning_times[0][i] = 0.0;
            learning_times[1][i] = 0.0;
        }
    }else{
        threads_keep_working = false;
        learning_batch_size = 1;
    }
    episode_n = int(5000 / log2(learning_batch_size));
    if(episode_n < 1000){
        episode_n = 1000;
    }
}

void DQN::resetAgents(int hidden_count,int hidden_size,int threads_number){
    update_local_agent = false;
    network_learned = false;
    learning_batch_size = threads_number;
    if(threads_number > 0){
        threads_keep_working = true;
        use_threads = true;
    }else{
        learning_batch_size = 1;
        threads_keep_working = false;
        use_threads = false;
    }
    int hc = hidden_count;
    if(hc <= 0){
        hc = 10;
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
    episode_n = int(5000 / log2(learning_batch_size));
    if(episode_n < 1000){
        episode_n = 1000;
    }
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
            memory.erase(memory.begin(), memory.begin() +( memory.size() / 3));
        }
    }
    
    memory.push_back(DQNMemoryUnit(game.getGameRepresentation(),oldGameRepresentation,action,fb.reward,fb.done));
    return fb.done;
}



void DQN::learn_from_memory(int thread_id){
    float q_correction=0, max=0;
    if(dev_debug_threading)
        cout<<"MAIN choose_random_from_memory"<<endl;
    DQNMemoryUnit learningExample = choose_random_from_memory();
    //cout<<"computeOutput thread:"<<thread_id<<endl;
    if(dev_debug_threading)
        cout<<"MAIN get max"<<endl;
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
    if(dev_debug_threading)
        cout<<"MAIN agent.learn"<<endl;
    agent.learn(q_correction,learningExample.action,learningExample.game,true);
}

std::chrono::_V2::system_clock::time_point DQN::collect_time(bool start_else_end,int thread_id,std::chrono::_V2::system_clock::time_point time_differ){
    using std::chrono::high_resolution_clock;
    using std::chrono::duration;
    std::chrono::_V2::system_clock::time_point action_time = high_resolution_clock::now();
    if(start_else_end){
        duration<long double, std::milli> t =  action_time - start_learning_time;
        learning_times[0][thread_id + 1] = t.count(); // true czas startu procesu
        
        //cout<<"colected time: "<<learning_times[0][learning_batch_size + 1]<<endl;
    }else{
        duration<long double, std::milli> t =  action_time - time_differ;
        learning_times[1][thread_id + 1] = t.count(); // false czas końca procesu
        //cout<<"colected time: "<<learning_times[1][learning_batch_size + 1]<<endl;
    }
    return action_time;
}

void DQN::safe_data_to_file(bool is_update_times){
    thread_times_file<<to_string(folder_to_safe_to)<<","<<to_string(learning_batch_size);
    if(is_update_times){
            thread_times_file<<",U,";
        }else{
            thread_times_file<<",L,";
        }
    for(int t_id =0; t_id < 128 + 1;t_id ++){
        thread_times_file<<learning_times[0][t_id]<<","<<learning_times[1][t_id]<<",";
    }
    thread_times_file<<endl;
}


void DQN::makeDQN_Thread(int thread_idx){
    if(dev_debug_threading)
        cout<<"started therad "<<thread_idx<<endl;
    Environment2D local_game;
    Policy local_agent = agent.copy();
    //Policy local_target_agent = target_agent.copy();
    DQNMemoryUnit learningExample;
    float q_correction=0, max=0;
    bool update_local_agent_flag = update_local_agent;

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
        update_local_agent_flag = update_local_agent;
        start_lck.unlock();
        if(dev_debug_threading)
            cout<<"therad "<<thread_idx<<" started next learning session"<<endl;

        std::chrono::_V2::system_clock::time_point tp_saved = collect_time(true,thread_idx); // zapisujemy czas rozpoczęcia uczenia

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
        finished_threaded_learning.notify_all();
        finish_lck.unlock();
        
        local_agent.change_weights(true);

        if(local_game.check_if_good_enougth(&local_agent)){
            final_agent = local_agent;
            network_learned = true;
            if(dev_debug_threading)
                cout<<"therad "<<thread_idx<<" HAVE A SOLUTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<endl;
            //showBestChoicesFor(local_agent);
        }
        
        collect_time(false,thread_idx,tp_saved); // zapisujemy czas zakończenia uczenia

        //czekamy aż Master powie że można zupdateować local agenta
        if(update_local_agent_flag){
            if(dev_debug_threading)
                cout<<"therad "<<thread_idx<<" whaitibng for weigths update approval (pre LOCK)"<<endl;
            start_ulck.lock(); 
            if(dev_debug_threading)
                cout<<"therad "<<thread_idx<<" whaitibng for weigths update approval (post LOCK)"<<endl;
            start_threaded_updateing.wait(start_ulck,[&]{
                return !thread_finished_updateing[thread_idx];
            });
            start_ulck.unlock();

            std::chrono::_V2::system_clock::time_point tp_saved = collect_time(true,thread_idx); // zapisujemy czas rozpoczęcia updateowania

            
            //lck.unlock();true== 0){
            local_agent.updateParameters(&agent);
            if(dev_debug_threading)
                cout<<"therad "<<thread_idx<<" updateing local_agent"<<endl;
            

            
            thread_finished_updateing[thread_idx] = true; // zaznaczamy że skończyliśmy udate
            collect_time(false,thread_idx,tp_saved); // zapisujemy czas zakończenia updateowania
            finish_ulck.lock();
            if(dev_debug_threading){
                cout<<"therad "<<thread_idx<<" pre notify all"<<endl;
            }
            finished_threaded_updateing.notify_all();
            if(dev_debug_threading){
                cout<<"therad "<<thread_idx<<" post notify all"<<endl;
            }
            finish_ulck.unlock();
        }
        if(dev_debug_threading)
            cout<<"therad "<<thread_idx<<" whaitibng for next learning session to start"<<endl;
    }
    if(dev_debug_threading)
        cout<<"therad "<<thread_idx<<" STOPED "<<endl;
}

Policy DQN::train(double* learning_time,int* steps_done,int* episodes){
    using std::chrono::high_resolution_clock;
    std::chrono::_V2::system_clock::time_point tp_saved;
    //thread_times_file.open("work_balance_data/"+std::to_string(folder_to_safe_to)+"/threadsTime_"+ std::to_string(learning_batch_size) + ".csv", std::ios::app);
    thread_times_file.open("work_balance_data/threadsTime.csv", std::ios::app);
    thread_times_file<<"started working"<<endl;;
    final_agent = agent.copy();
    assert(learning_batch_size <= threads_numer);
    bool whait_for_update = false;
    int done_counter = 0;
    if(show_output)
        cout << "Start training,  episodes:"<<episode_n<<endl;
    auto start_time = chrono::steady_clock::now();
    int i;
    
    int update_countdown = update_local_agent_frequency;
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
            thread_finished_updateing[thread_id] = true;
            threads[thread_id] = std::thread([this,thread_id](){this->makeDQN_Thread(thread_id);});
        }
    start_learning_time = high_resolution_clock::now();
    for (i=0 ; i<episode_n ; i++){
        steps=0;
        done=false;
        tp_saved = collect_time(true,-1); // zapisujemy czas o starcie pracy głównego wątku 
        game.reset();
        while(!done && steps<n_steps_in_one_go && network_learned == false){
            //cout<<"start sampling"<<endl
            if(dev_debug_threading)
                cout<<"MAIN_TRAIN collecting memory steps, steps:"<<steps<<endl;
            if(make_only_one_learning_steps_ALWAYS){
                steps++;
                done = collect_memory_step();
            }else{
                for(int b = 0; b<learning_batch_size && done == false; b++){
                    steps++;
                    done = collect_memory_step();
                }
            }
            collect_time(false,-1,tp_saved); // zapisujemy czas o końcu pracy głównego wątku 

            if(use_threads){
                
                if(dev_debug_threading)
                    cout<<"MAIN_TRAIN inform that threads can start lerning"<<endl;
                s_lck.lock();// ustawiamy flagi uczenia 
                for(int b = 0; b<learning_batch_size  && b<memory.size(); b++){
                    thread_finished_learning[b] = false;
                    if(dev_debug_threading)
                        cout<<"MAIN_TRAIN inform that thread:"<<b<<" can start lerning:"<<!thread_finished_learning[b]<<endl;
                }
                if(update_countdown == 0){// steting up what should happend after learning
                    update_countdown = update_local_agent_frequency;
                    update_local_agent = true;
                }else{
                    update_countdown--;
                }
                start_threaded_learning.notify_all();// powiadamiamy wszystkie wątki że można uczyć
                s_lck.unlock();

                if(dev_debug_threading)
                    cout<<"MAIN_TRAIN started whaiting for threaded learning to end"<<endl;
                f_lck.lock();
                finished_threaded_learning.wait(f_lck,[&](){
                    for(int b = 0; b<learning_batch_size  && b<memory.size(); b++){
                        if(thread_finished_learning[b] == false){
                            if(dev_debug_threading)
                                cout<<"MAIN_TRAIN thread "<<b<<" didnt finish learning"<<endl;
                            return false;
                        }
                    }
                    if(dev_debug_threading)
                        cout<<"MAIN_TRAIN learning finished notified"<<endl;
                    return true;
                });
                f_lck.unlock();

                //TODO zapisujemy do pliku ile czasu wyniosła nas główna pętla
                safe_data_to_file(false);

                tp_saved = collect_time(true,-1);// zapisujemy czas początku prac main thread przy update lub czas startu zbierania tracea
                
                if(update_local_agent){
                    if(dev_debug_threading)
                        cout<<"MAIN_TRAIN inform we can start updateing (PRE LOCK)"<<endl;
                    for(int b = 0; b<learning_batch_size  && b<memory.size(); b++){
                        thread_finished_updateing[b] = false;// ustawiamy flagi updateu 
                    }
                    start_lck.lock();
                    if(dev_debug_threading)
                        cout<<"MAIN_TRAIN inform we can start updateing (POST LOCK)"<<endl;
                    // powiadamiamy wszystkie wątki że można zupdateować swojego local agenta
                    start_threaded_updateing.notify_all();// powiadamiamy wszystkie wątki że można zupdateować swojego local agenta
                    start_lck.unlock();
                    

                    if(game.check_if_good_enougth(&agent)){
                        network_learned = true;
                        final_agent = agent;
                        if(dev_debug_threading)
                            cout<<"MAIN HAVE A SOLUTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<endl;
                        //showBestChoicesFor(local_agent);
                    }

                    collect_time(false,-1,tp_saved);// koniec robót main thread podczas updateowania agenta

                    if(dev_debug_threading)
                        cout<<"MAIN_TRAIN w8ing for threads update  (PRE LOCK)"<<endl;
                    finish_lck.lock();// czekamy na koniec updateowania agenta
                    if(dev_debug_threading)
                        cout<<"MAIN_TRAIN w8ing for threads update  (POST LOCK)"<<endl;
                    finished_threaded_updateing.wait(finish_lck,[&](){
                        if(dev_debug_threading)
                            cout<<"MAIN_TRAIN checking if threads ended"<<endl;
                        for(int b = 0; b<learning_batch_size  && b<memory.size(); b++){
                            if(thread_finished_updateing[b] == false){
                                if(dev_debug_threading)
                                    cout<<"MAIN_TRAIN thread "<<b<<" didnt finish"<<endl;
                                return false;
                            }
                        }
                        if(dev_debug_threading)
                            cout<<"MAIN_TRAIN ALL THREADS ENDED"<<endl;
                        return true;
                    });
                    finish_lck.unlock();
                    update_local_agent = false;
                    if(dev_debug_threading)
                        cout<<"MAIN_TRAIN post w8ing for update END"<<endl;


                //TODO zapisujemy do pliku ile czasu wyniosła nas główna pętla
                safe_data_to_file(true);

                collect_time(true,-1);// zapisujemy czas początku prac main thread przy głównej pentli (zbieraniu tracea)
                    
                }else{
                    if(game.check_if_good_enougth(&agent)){
                        network_learned = true;
                        final_agent = agent;
                        if(dev_debug_threading)
                            cout<<"MAIN HAVE A SOLUTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<endl;
                        //showBestChoicesFor(local_agent);
                    }
                }

                
                
            }else{
                if(dev_debug_threading)
                        cout<<"MAIN learn_from_memory"<<endl;
                learn_from_memory(-1);   
                if(game.check_if_good_enougth(&agent)){
                    network_learned = true;
                    final_agent = agent;
                    if(dev_debug_threading)
                        cout<<"MAIN HAVE A SOLUTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<endl;
                    //showBestChoicesFor(local_agent);
                } 
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

        if(network_learned || have_any_nan(agent)){
            break;
        }
    }
    {// killing childs section
        threads_keep_working = false;
        for(int b = 0; b<learning_batch_size  && b<memory.size(); b++){
            thread_finished_learning[b] = true;
        }
        start_lck.lock();
        start_threaded_updateing.notify_all();
        start_lck.unlock();
        std::unique_lock<std::mutex> start_lck(start_threaded_learning_mtx);// ustawiamy flagi uczenia 
        for(int b = 0; b<learning_batch_size  && b<memory.size(); b++){
            thread_finished_learning[b] = false;
        }
        update_local_agent = false;
        start_threaded_learning.notify_all();// powiadamiamy wszystkie wątki że można uczyć
        start_lck.unlock();
        for(int b = 0; b<learning_batch_size  && b<memory.size(); b++){
                if (threads[b].joinable()){
                    threads[b].join();
                }
        }
    }

    if(network_learned){
        exec_time = chrono::steady_clock::now() - start_time;
    }else{
        exec_time = start_time - chrono::steady_clock::now();
    }
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

    thread_times_file.close();
    
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

bool DQN::have_any_nan(Policy agent){ 
    int action = 0;
    for(int h = 0;h<game.lengthH || h<game.lengthH ;h++){
        agent.computeOutput_fast(game.toGameRepresentation(h,h)).getMax( NULL, &action,NULL);
        if(isnan(action)){
            return true;
        }
    }
    return false;
}

//TODO

// dodać system anti-corupted (* wywalamy czas uczenia na inf/-1/9999999 i zwracamy pierwotną postać sieci gdy otrzymujemy nany na wyjściu )

// nie ma co updateować agenta w wątkach trzeba jedynie przekazać pochodne po W/B i main to sobie zupdateuje. ???

// main nie powinien czekać aż wątki skończą updateować, przystepujemy od razu do zbieranie stepsów ? 

// dodać nową metodę updateowania local agenta !!!! ??? myślałem żeby każdy wątek uczył swojego przez x okrążeń my w tym czasie zbieramy tracy i w pewnym momencie stopujemy i synchronizujemy globalnego agenmta względem tego co ejst na poszczególnych wątkach
void collect_time(bool start_else_end,int thread_id);