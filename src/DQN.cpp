#include "../include/DQN.h"

DQN::DQN(){
    learning_rate = 0.005;
    gamma = 0.8;
    eps = 1.0; // procent określający z jakim prawdopodobieństwem wykonamy ruch losowo
    epsDecay = 0.85; // procent maleje TODO
    target_agent_update_freaquency = 50;
    target_agent_count_down = target_agent_update_freaquency;
    n_steps_in_one_go = 10 * game.length();
    episode_n = 1000;
    learning_batch_size = 1;

    agent = Policy(game.length(), 10,8, game.actionsCount, learning_rate);
    target_agent = agent.copy();
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
    memory.push_back(DQNMemoryUnit(game.getGameRepresentation(),oldGameRepresentation,action,fb.reward,fb.done));
    return fb.done;
}

void DQN::learn_from_memory(bool update_on_spot){
    DQNMemoryUnit learningExample = choose_random_from_memory();
    float q_correction=0, max=0;
    if(use_target_agent){
        target_agent.computeOutput(learningExample.game_next).getMax( NULL, NULL, &max);
    }else{
        agent.computeOutput(learningExample.game_next).getMax( NULL, NULL, &max);
    }
    if(learningExample.done == true){
        q_correction = learningExample.reward;
    }else{
        q_correction = learningExample.reward + gamma*max;
    }

    agent.learn(q_correction,learningExample.action,learningExample.game,update_on_spot);
}

Policy DQN::train(double* learning_time,int* steps_done,int* episodes){
    int done_counter = 0;
    if(show_output)
        cout << "Start training,  episodes:"<<episode_n<<endl;
    auto start_time = chrono::steady_clock::now();
    int i;
    for (i=0 ; i<episode_n ; i++){
        int steps=0;
        bool done=false;
        
        game.reset();
        while(!done && steps<n_steps_in_one_go){
            for(int b = 0; b<learning_batch_size && done == false; b++){
                steps++;
                done = collect_memory_step();
            }
            for(int b = 0; b<learning_batch_size ; b++){
                learn_from_memory(b == learning_batch_size -1);
                if(use_target_agent)
                    if(target_agent_count_down == 0){
                        target_agent.updateParameters(agent);
                        target_agent_count_down = target_agent_update_freaquency;
                    }else{
                        target_agent_count_down--;
                    }
            }
            if(done|| steps>=n_steps_in_one_go){
                done_counter += steps;
                eps *= epsDecay;
                if(use_target_agent){
                    target_agent.updateParameters(agent);
                    target_agent_count_down = target_agent_update_freaquency;
                }
            }
        }


        if(show_output){
            cout << "Episode " << i+1 << "/" << episode_n << "\t";
            cout << "[" << steps << " steps] eps:"<< eps << endl ;//<<endl << " Szansa ma losowy krok" << eps*100.0<<endl;
        }

        if((steps == n_steps_in_one_go && eps < 0.01) || (eps < 0.001)){ // reseting exploration chance
            eps = 1.0;
        }
        if(show_output){
            showBestChoicesFor(agent);
        }

        if(game.check_if_good_enougth(agent,show_output)){
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