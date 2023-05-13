RFLAGS = -Ofast -std=c99 -lm -lpthread -lSDL2
DFLAGS = -O1 -std=c99 -Wall -Wextra -g -lm -lSDL2
FILES = main.c
NAME = neuralnetwork

make :
	mkdir -p ./build/debug/
	gcc $(FILES) $(DFLAGS) -o ./build/debug/$(NAME)
release :
	mkdir -p ./build/release/
	gcc $(FILES) $(RFLAGS) -o ./build/release/$(NAME)


