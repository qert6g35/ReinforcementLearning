
CC = g++
ODIR = obj
PROG = main
CXXFLAGS = -std=c++11

OBJS = $(ODIR)/main.o $(ODIR)/matrix.o
$(PROG) : $(ODIR) $(OBJS)
	$(CC) -o $@ $(OBJS) $(CXXFLAGS)

$(ODIR)/main.o : ./src/main.cpp ./include/matrix.h ./include/environment.h
	$(CC) -c ./src/main.cpp -o $@ $(CXXFLAGS)

$(ODIR)/matrix.o : ./src/matrix.cpp ./include/matrix.h
	$(CC) -c ./src/matrix.cpp -o $@ $(CXXFLAGS)

$(ODIR) :
	mkdir -f $(ODIR)

.PHONY: clean

clean:
	@echo %PATH%
	rm -Rf ./$(ODIR)
	rm -Rf ./$(PROG)
