#include "../include/DQN.h"

DQN::DQN(){
    learning_rate = 0.005;
    gamma = 0.8;
    eps = 1.0; // procent określający z jakim prawdopodobieństwem wykonamy ruch losowo
    epsDecay = 0.95; // procent maleje TODO
    target_agent_update_freaquency = 50;
    target_agent_count_down = target_agent_update_freaquency;
    n_steps_in_one_go = 10 * game.length();
    episode_n = 1000;
    learning_batch_size = 5;
    
    agent = Policy(game.length(), 10,8, game.actionsCount, learning_rate,threads_numer);
    target_agent = agent.copy();
    //visited = vector<bool>(game.length(),false);
}

void DQN::resetAgents(int hidden_count,int hidden_size){
    int hc = hidden_count;
    if(hc <= 0){
        hc = 8;
    }
    int hs = hidden_size;
    if(hs <= 0){
        hidden_size = 10;
    }
    n_steps_in_one_go = 10 * game.length();
    agent = Policy(game.length(), hs,hc, game.actionsCount, learning_rate,threads_numer);
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
        // if(show_output)
        //     cout<<"pick_random_action"<<endl;
        action = rand()%game.actionsCount;
    }
    else{
        //picking action that follows Q-table
        // if(show_output)
        //     cout<<"pick_action"<<endl;
        agent.computeOutput(oldGameRepresentation).getMax( NULL, &action, NULL);
    }

    // take action in enviroment
    //Qaprox.print(cout);
    Observation fb = game.step(action);
    if(!use_memory){
        memory.clear();
    }
    // else{
    //     // if(memory.size() > max_memory_size){
    //     //     cout<<"memory precut-size:"<<memory.size()<<endl;
    //     //     memory.erase(memory.begin(), memory.begin() +( memory.size() / 2));
    //     //     cout<<"memory postcut-size:"<<memory.size()<<endl;
    //     // }
    // }
    memory.push_back(DQNMemoryUnit(game.getGameRepresentation(),oldGameRepresentation,action,fb.reward,fb.done));
    return fb.done;
}

void makeThreadLearn(int thread_idx,Policy &agent,float q_correction,DQNMemoryUnit learningExample,mutex &mtxW,mutex &mtxB){//(Policy &useAgent,float q_correction,DQNMemoryUnit memory_unit,int thread_id){//,std::mutex*safe_to_global_dJdW,std::mutex*&safe_to_global_dJdB){
    //std::cout<<"thread is alive"<<endl<<"and we have policy"<<endl;
    agent.learn_thread(q_correction,learningExample.action,learningExample.game,thread_idx,mtxW,mtxB);
    //useAgent.learn_thread(q_correction,memory_unit.action,memory_unit.game,thread_id,NULL,NULL);
}

void DQN::learn_from_memory(int thread_id){
    //cout<<"choose_random_from_memory thread:"<<thread_id<<endl;
    DQNMemoryUnit learningExample = choose_random_from_memory();
    float q_correction=0, max=0;
    //cout<<"computeOutput thread:"<<thread_id<<endl;
    if(use_target_agent){
        target_agent.computeOutput(learningExample.game_next).getMax( NULL, NULL, &max);
    }else{
        agent.computeOutput(learningExample.game_next).getMax( NULL, NULL, &max);
    }
    //cout<<"q_correction thread:"<<thread_id<<endl;
    if(learningExample.done == true){
        q_correction = learningExample.reward;
    }else{
        q_correction = learningExample.reward + gamma*max;
    }
    if(thread_id >= 0){
        //std::cout<<"starting thread:"<<thread_id<<endl;
        threads[thread_id] = std::thread( std::bind(makeThreadLearn,thread_id,std::ref(agent),q_correction,learningExample,std::ref(safe_to_global_dJdW),std::ref(safe_to_global_dJdB)) );
    }else{
        //std::cout<<"standard learning thread:"<<thread_id<<endl;
        agent.learn(q_correction,learningExample.action,learningExample.game);
    }
}

Policy DQN::train(double* learning_time,int* steps_done,int* episodes){
    assert(learning_batch_size <= threads_numer);
    int done_counter = 0;
    if(show_output)
        cout << "Start training,  episodes:"<<episode_n<<endl;
    auto start_time = chrono::steady_clock::now();
    int i;
    bool network_learned = false;
    for (i=0 ; i<episode_n ; i++){
        int steps=0;
        bool done=false;
        
        game.reset();
        while(!done && steps<n_steps_in_one_go && network_learned == false){
            //cout<<"start sampling"<<endl;
            //for(int b = 0; b<learning_batch_size && done == false; b++){
                steps++;
                done = collect_memory_step();
            //}
            //cout<<"start learning"<<endl;
            for(int b = 0; b<learning_batch_size - 1 && b<memory.size() - 1; b++){
                //cout<<"starting learn_from_memory:"<<b<<endl;
                learn_from_memory(b);    
            }
            for(int b = 0; b<learning_batch_size - 1 && b<memory.size() - 1; b++){
                //cout<<"joining thread:"<<b<<endl;
                threads[b].join();   
            }
            learn_from_memory(-1);
            if(use_target_agent)
                if(target_agent_count_down == 0){
                    target_agent.updateParameters(agent);
                    target_agent_count_down = target_agent_update_freaquency;
                    network_learned = game.check_if_good_enougth(agent);
                }else{
                    target_agent_count_down--;
                    eps *= epsDecay;
                }
            if(done|| steps>=n_steps_in_one_go){
                done_counter += steps;
                network_learned = game.check_if_good_enougth(agent);
                if(use_target_agent){
                    target_agent.updateParameters(agent);
                    target_agent_count_down = target_agent_update_freaquency;
                }
            }


            if((steps == n_steps_in_one_go && eps < 0.01) || (eps < 0.001)){ // reseting exploration chance
                eps = 1.0;
            }
        }


        if(show_output){
            cout << "Episode " << i+1 << "/" << episode_n << "\t";
            cout << "[" << steps << " steps] eps:"<< eps << endl ;//<<endl << " Szansa ma losowy krok" << eps*100.0<<endl;
        }

        //if((steps == n_steps_in_one_go && eps < 0.01) || (eps < 0.001)){ // reseting exploration chance
        //    eps = 1.0;
        //}
        if(show_output){
            showBestChoicesFor(agent);
        }

        if(network_learned){
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
    return agent;
}

void DQN::showBestChoicesFor(Policy agent){ 
    for(int h = 0;h<game.lengthH;h++){
        for(int w = 0;w<game.lengthW;w++){
            int action = 0;
            agent.computeOutput(game.toGameRepresentation(h,w)).getMax( NULL, &action, NULL);
            if(action==0){
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
    if(use_memory)
        return memory[(int)(rand() % memory.size())];
    return memory.back();
}

// the cat code

// 0 0 1 1  0  0  1  0
// 0 1 0 0  1  1  0  0
// 1 2 4 8 16 32 64 128



// 2+16+32 = 50 

// 4+8+64 = 76


// 5076

// 7650