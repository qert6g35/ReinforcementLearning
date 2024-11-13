
CC = g++
ODIR = obj
PROG = main
CXXFLAGS = -std=c++11

OBJS = $(ODIR)/main.o $(ODIR)/matrix.o $(ODIR)/policy.o #<< przy dodawaniu tutaj też musimy dodać jakie .o chcemy stworzyć 
$(PROG) : $(ODIR) $(OBJS)
	$(CC) -o $@ $(OBJS) $(CXXFLAGS)

$(ODIR)/main.o : ./src/main.cpp ./include/matrix.h ./include/environment.h ./include/policy.h
	$(CC) -c ./src/main.cpp -o $@ $(CXXFLAGS)

$(ODIR)/matrix.o : ./src/matrix.cpp ./include/matrix.h
	$(CC) -c ./src/matrix.cpp -o $@ $(CXXFLAGS)

$(ODIR)/policy.o : ./src/policy.cpp ./include/policy.h ./include/matrix.h
	$(CC) -c ./src/policy.cpp -o $@ $(CXXFLAGS)

$(ODIR) :
	mkdir $(ODIR)

.PHONY: clean

clean:
	rm -Rf ./$(ODIR)
	rm -Rf ./$(PROG)
