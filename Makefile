CC = g++
ODIR = obj
PROG = main
CXXFLAGS = -std=c++17 -O2
FLAGS = -ggdb 

OBJS = $(ODIR)/main.o $(ODIR)/DQN.o $(ODIR)/matrix.o $(ODIR)/policy.o  #<< przy dodawaniu tutaj też musimy dodać jakie .o chcemy stworzyć 
$(PROG) : $(ODIR) $(OBJS)
	$(CC) ${FLAGS} -o $@ $(OBJS) $(CXXFLAGS)

$(ODIR)/main.o : ./src/main.cpp ./include/DQN.h ./include/policy.h ./include/environment.h
	$(CC) ${FLAGS} -c ./src/main.cpp -o $@ $(CXXFLAGS)

$(ODIR)/matrix.o : ./src/matrix.cpp ./include/matrix.h
	$(CC) ${FLAGS} -c ./src/matrix.cpp -o $@ $(CXXFLAGS)

$(ODIR)/policy.o : ./src/policy.cpp ./include/policy.h ./include/matrix.h
	$(CC) ${FLAGS} -c ./src/policy.cpp -o $@ $(CXXFLAGS)

$(ODIR)/DQN.o : ./src/DQN.cpp ./include/DQN.h ./include/matrix.h ./include/environment.h ./include/policy.h
	$(CC) ${FLAGS} -c ./src/DQN.cpp -o $@ $(CXXFLAGS)

$(ODIR) :
	mkdir $(ODIR)

.PHONY: clean

clean:
	rm -Rf ./$(ODIR)
	rm -Rf ./$(PROG)
