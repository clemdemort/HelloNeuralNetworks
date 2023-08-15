#name of the executable
NAME = neuralnetwork
#files necessary for compilation
FILES = main.c src/pngloader.c
#compiler flags
LFLAGS = -lm -lpng -lpthread -lSDL2
RFLAGS = -Ofast -std=c11 $(LFLAGS)					#release
DFLAGS = -O1 -std=c11 -Wall -Wextra -g $(LFLAGS)	#debug

#by default build in debug
make :
	mkdir -p ./build/debug/
	gcc $(FILES) $(DFLAGS) -o ./build/debug/$(NAME)
release :
	mkdir -p ./build/release/
	gcc $(FILES) $(RFLAGS) -o ./build/release/$(NAME)


