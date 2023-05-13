#name of the executable
NAME = neuralnetwork
#files necessary for compilation
FILES = main.c src/matrix.c
#compiler flags
RFLAGS = -Ofast -std=c99 -lm -lpthread -lSDL2				#release
DFLAGS = -O1 -std=c99 -Wall -Wextra -g -lm -lpthread -lSDL2	#debug

#by default build in debug
make :
	mkdir -p ./build/debug/
	gcc $(FILES) $(DFLAGS) -o ./build/debug/$(NAME)
release :
	mkdir -p ./build/release/
	gcc $(FILES) $(RFLAGS) -o ./build/release/$(NAME)


