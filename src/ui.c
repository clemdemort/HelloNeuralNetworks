#include "ui.h"
#include <unistd.h>
#include <SDL2/SDL_render.h>
#include <SDL2/SDL_video.h>

pthread_t ui_thread;
char UI_RUNNING;

void * ui(void * arg){
    SDL_Window * window = SDL_CreateWindow("Hello Neural Networks", 0, 0,600 ,400, SDL_WINDOW_RESIZABLE);
    SDL_Renderer * renderer = SDL_CreateRenderer(window, -1, 0);
    UI_RUNNING = 1;
    while(UI_RUNNING){
        SDL_Rect r;
	    r.x = 10;
	    r.y = 10;
	    r.w = 20;
	    r.h = 20;
	    SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
	    SDL_RenderFillRect(renderer, &r);
        SDL_RenderPresent(renderer);
        usleep(16000);
    }
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
}

void init_ui_thread(){
    pthread_create(&ui_thread, NULL, &ui, NULL);
}

void join_ui_thread(){
    UI_RUNNING = 0;
    pthread_join(ui_thread, NULL);
}