/*
                Media.h
    a standalone header only library to simplify SDL2 calls
        author: clement bartolone

*/

#ifndef MEDIA_H
#define MEDIA_H
#define _POSIX_C_SOURCE 199309L
#include <SDL2/SDL_pixels.h>
#include <SDL2/SDL_render.h>
#include <bits/types/clock_t.h>
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <SDL2/SDL.h>
#include <time.h>


//Redefining some variable types for ease of use
typedef float    f32;
typedef double   f64;
typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int8_t   i8;
typedef int16_t  i16;
typedef int32_t  i32;
typedef int64_t  i64;
typedef size_t   usize;
typedef ssize_t  isize;
typedef unsigned char uc;

typedef struct MEDIA_H_CONTEXT{
    SDL_Window * window;        //you might need to pass this around
    SDL_Renderer * renderer;    //you might need to pass this around
    SDL_Texture * screenbuffer; //don't touch unless you know what you're doing
    u32 *pixels;                //allows you to edit individual pixels
    const Uint8* keystates;     //don't touch unless you know what you're doing
    int RUNNING;                //specifies if the media loop is running
    float elapsedTime;          //the elapsedTime, handy for physics
    u32 scrW,scrH,scale;
}MEDIA_H_CONTEXT;

typedef struct rgba {
	uc R;
	uc G;
	uc B;
	uc A;
}rgba;

//*******************//
// Utility Functions //
//*******************//

//simple function to make declaring colours faster
rgba colour(uc R,uc G,uc B,uc A){
	rgba col = {R,G,B,A};
	return col;
}

//hopefully this is cheap though there are 4 casts
u32 colToPixel(rgba col){
    u32 res  = (u32)col.A;
        res += (u32)col.B*256;
        res += (u32)col.G*256*256;
        res += (u32)col.R*256*256*256;
    return res;
}


//*****************************************//
// basic functions to manage an SDL context//
//*****************************************//

MEDIA_H_CONTEXT newcontext(char * title, int xpos, int ypos, int width, int height, int flags){
    MEDIA_H_CONTEXT context;
    context.window = SDL_CreateWindow(title, xpos, ypos,width ,height, flags);
    context.renderer = SDL_CreateRenderer(context.window, -1, SDL_RENDERER_ACCELERATED);
    context.RUNNING = 1;
    context.scale = 1;
    context.pixels = malloc(width * height * 4);
    return context;
}

void destroycontext(MEDIA_H_CONTEXT context){
    free(context.pixels);
    SDL_DestroyRenderer(context.renderer);
    SDL_DestroyWindow(context.window);
}

void handle_events(MEDIA_H_CONTEXT * context){
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                context->RUNNING = 0;
            }
        }
    context->keystates = SDL_GetKeyboardState(NULL);
	SDL_PumpEvents();
}

void update_MEDIA_H_CONTEXT(MEDIA_H_CONTEXT * context,void (*update_function)(MEDIA_H_CONTEXT*)){
    static struct timespec t1, t2;
    t1 = t2;
    clock_gettime(CLOCK_REALTIME, &t2);
    long int t = (t2.tv_nsec - t1.tv_nsec)/1000;
    if(t > 0){
        context->elapsedTime = (double)t/1000000.0;
    }

    static u32 w,h;
    u32 w2,h2;
    GetWindowSize(context,&w2, &h2);
    if(w != w2 || h != h2){//resize
        free(context->pixels);
        context->pixels = calloc(w2 * h2, 4);
    }

    h = h2;
    w = w2;
    h2 = h/context->scale;
    w2 = w/context->scale;
    context->scrH = h2;
    context->scrW = w2;
    context->screenbuffer = SDL_CreateTexture(context->renderer,SDL_PIXELFORMAT_RGBA8888,SDL_TEXTUREACCESS_STREAMING,w2,h2);


    handle_events(context);
    update_function(context);

    void *px;
    int pitch;
    SDL_LockTexture(context->screenbuffer, NULL, &px, &pitch);
    {
        for (u32 y = 0; y < h2; y++) {
            memcpy(
                &((u8*) px)[y * pitch],
                &context->pixels[y * w2],
                w2 * 4);
        }
    }
    SDL_UnlockTexture(context->screenbuffer);
    SDL_SetRenderTarget(context->renderer, NULL);
    SDL_SetRenderDrawColor(context->renderer, 0, 0, 0, 0xFF);
    SDL_SetRenderDrawBlendMode(context->renderer, SDL_BLENDMODE_NONE);

    SDL_RenderClear(context->renderer);
    SDL_RenderCopyEx(
        context->renderer,
        context->screenbuffer,
        NULL,
        NULL,
        0.0,
        NULL,
        0);

    SDL_RenderPresent(context->renderer);
    SDL_DestroyTexture(context->screenbuffer);
    //free(context->pixels);
}

//will return 1 if key is currently being pressed
int GetKey(MEDIA_H_CONTEXT * context,int scancode){
	return context->keystates[scancode];
}

void GetMouseState(int * left,int * right){
	int buttons = SDL_GetMouseState(NULL, NULL);
	if ((buttons & SDL_BUTTON_LMASK) != 0) {
		*left = 1;
	}else *left = 0;
	if ((buttons & SDL_BUTTON_RMASK) != 0) {
		*right = 1;
	}else *right = 0;

}

void GetMousePos(int * X,int * Y){
	int buttons = SDL_GetMouseState(X,Y);
}

void GetWindowSize(MEDIA_H_CONTEXT * context,int * width,int * height){
	SDL_GetRendererOutputSize(context->renderer,width,height);
}

//******************//
// Drawing functions//
//******************//

void ChangeTitle(MEDIA_H_CONTEXT * context,char * title){
	SDL_SetWindowTitle(context->window, title);
}

void RenderClear(MEDIA_H_CONTEXT * context,rgba col){
	SDL_SetRenderDrawColor(context->renderer, col.R, col.G, col.B, col.A);
	SDL_RenderClear(context->renderer);
}
void RenderPixel(MEDIA_H_CONTEXT * context,rgba col, int x, int y){
    context->pixels[y*context->scrW + x] = colToPixel(col);
}

void setPixelValue(MEDIA_H_CONTEXT * context,u32 col, int x, int y){
    context->pixels[y*context->scrW + x] = col;
}

void RenderRectangle(MEDIA_H_CONTEXT * context,rgba col,long int x, long int y ,unsigned long int sx,unsigned long int sy){
	SDL_Rect r;
	r.x = x;
	r.y = y;
	r.w = sx;
	r.h = sy;
	SDL_SetRenderDrawColor(context->renderer, col.R, col.G, col.B, col.A);
	SDL_RenderFillRect(context->renderer, &r);
}

void RenderCircleOutline(MEDIA_H_CONTEXT * context,rgba col, int x, int y, int r){
	u32 c = colToPixel(col);
	const int32_t diameter = (r * 2);

   int32_t x2 = (r - 1);
   int32_t y2 = 0;
   int32_t tx = 1;
   int32_t ty = 1;
   int32_t error = (tx - diameter);

   while (x2 >= y2)
   {
      //  Each of the following renders an octant of the circle
      setPixelValue(context,c, x + x2, y - y2);
      setPixelValue(context,c, x + x2, y + y2);
      setPixelValue(context,c, x - x2, y - y2);
      setPixelValue(context,c, x - x2, y + y2);
      setPixelValue(context,c, x + y2, y - x2);
      setPixelValue(context,c, x + y2, y + x2);
      setPixelValue(context,c, x - y2, y - x2);
      setPixelValue(context,c, x - y2, y + x2);

      if (error <= 0)
      {
         ++y2;
         error += ty;
         ty += 2;
      }

      if (error > 0)
      {
         --x2;
         tx += 2;
         error += (tx - diameter);
      }
   }
}

void RenderCircle(MEDIA_H_CONTEXT * context,rgba col, int x, int y, int r){
	u32 c = colToPixel(col);
    if(r == 0){RenderPixel(context,col,x,y);goto end;}
    for (int w = 0; w < r * 2; w++)
        for (int h = 0; h < r * 2; h++)
        {
            int dx = r - w; // horizontal offset
            int dy = r - h; // vertical offset
            if ((dx*dx + dy*dy) <= (r * r))
            {
                setPixelValue(context,c, x + dx,  y + dy);
            }
        }
    end:
}

#endif
