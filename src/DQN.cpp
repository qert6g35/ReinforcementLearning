#include "../include/DQN.h"

DQN::DQN(){
    agent = Policy(game.length(), 10,8, game.actionsCount, 0.01);
    target_agent = agent.copy();

    gamma = 0.8;
    eps = 1.0; // procent określający z jakim prawdopodobieństwem wykonamy ruch losowo
    epsDecay = 0.85; // procent maleje TODO
    target_agent_update_freaquency = 25;
    target_agent_count_down = target_agent_update_freaquency;
    n_steps_in_one_go = 500;
    episode_n = 2000;
}

Policy DQN::train(){
    int done_counter = 0;
    //cout <<"agent        "<< odp1 <<"target_agent "<< odp2 << endl;
    cout << "Start training,  episodes:"<<episode_n<<endl;
    auto start_time = chrono::steady_clock::now();
    for (int i=0 ; i<episode_n ; i++){
        int steps=0, maxIndex=0, action=0;
        double q_correction=0, max=0;
        bool done=false;
        Matrix Qaprox;
        
        game.reset();

        int stepper = 0;

        done=false;
        while(!done && steps<n_steps_in_one_go){
            steps++;
            // save enviroment before takeing action
            std::vector<double> oldGameRepresentation = game.getGameRepresentation();

            // take random actions sometimes to allow game exploration
            if(((double) rand() / RAND_MAX) < eps){
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
            memory.push_back(DQNMemoryUnit(game.getGameRepresentation(),oldGameRepresentation,action,fb.reward,fb.done));

            done = fb.done;
            // if(done){
            //     cout<<"GAME END pozH:"<<game.positionH<<" pozW:"<<game.positionW<<endl;
            //     //showBestChoicesFor(agent);
            // }
            //
            // saveing that moment in 
            // if(show_output)
            //     cout<<"adding to memory"<<endl;


    
            //DQNMemoryUnit learningEgxample = memory[memory.size() - 1];//choose_random_from_(memory,gen);
            DQNMemoryUnit learningEgxample = choose_random_from_();

            //DQNMemoryUnit learningEgxample = DQNMemoryUnit(game.getGameRepresentation(),oldGameRepresentation,action,fb.reward,done);

            // get best action in next state
            Matrix Qprox_next = target_agent.computeOutput(learningEgxample.game_next);
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
            // if(show_output)
            //     cout<<"corection "<<q_correction<<" on action:"<< learningEgxample.action<<endl;
            agent.learn(q_correction,learningEgxample.action,learningEgxample.game);
            // if(show_output)
            //     cout<<"post correction"<<endl;
            //agent.learn(10,0,learningEgxample.game);

            if(target_agent_count_down == 0){
                eps *= epsDecay;
                target_agent.updateParameters(agent);
                // if(agent.computeOutput(game.toGameRepresentation(0,0,game.lengthW,game.lengthH)).haveAnyNan()){
                //     agent.updateParameters(target_agent);
                // }else{
                //     if(show_output)
                //         cout<<"update target_agent"<<endl;
                //     target_agent.updateParameters(agent);
                // }
                target_agent_count_down = target_agent_update_freaquency;
            }else{
                target_agent_count_down--;
            }
        }
        if(show_output){
            cout << "Episode " << i+1 << "/" << episode_n << "\t";
            cout << "[" << steps << " steps] eps:"<< eps << endl ;//<<endl << " Szansa ma losowy krok" << eps*100.0<<endl;
        }
        target_agent.updateParameters(agent); // update target agent after every batch
        target_agent_count_down = target_agent_update_freaquency;

        if((steps == n_steps_in_one_go && eps < 0.01) || (eps < 0.001)){ // reseting exploration chance
            eps = 1.0;
        }
        if(show_output){
            showBestChoicesFor(target_agent);
        }
        steps = 0;
        done = false;
        game.reset();
        while(!done){
            stepper++;
            Matrix actions = target_agent.computeOutput({game.getGameRepresentation()});
            actions.getMax( NULL, &action, NULL);
            Observation fb = game.step(action);
            done = fb.done;
            if(stepper > game.length()+1){
                if(show_output)
                    cout<<"agent cant end game on its own"<<endl;
                break;
            }
        }
        //eraseLines(game.lengthH+1);

        if(done == true){
            break;
        }
    }
    auto end_time = chrono::steady_clock::now();
    exec_time = end_time - start_time;
    //odp1 = agent.computeOutput({game.getGameRepresentation()});
    //odp2 = target_agent.computeOutput({game.getGameRepresentation()});
    //cout <<"agent        "<< odp1 <<"target_agent "<< odp2 << endl;
    cout<<" Training took:"<<exec_time.count() / 1000.0<<"s"<<endl;
    return target_agent;
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

DQNMemoryUnit DQN::choose_random_from_(){ 
    return memory[(int)(rand() % memory.size())];
}