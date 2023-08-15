#include "src/media.h"
#include "src/neuralLib.h"
#include "src/pngloader.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>

model mdl;//model
uc PAUSE;
img test;

nlf distxy(nlf x, nlf y , nlf cx, nlf cy){
	return sqrt(((cx-x)*(cx-x)) + ((cy-y)*(cy-y)) );
}

void draw_out(MEDIA_H_CONTEXT * context,model nn){
	vec vin = newVec(2);
	descriptor D = getDescriptor(nn);
	activations act = newActivations(D);
	destroyDesc(D);
	nlu w = context->scrW;
	nlu h = context->scrH;
	nlu a = 0;
	for(nlu y = 0; y < h;y++)
		for(nlu x = 0; x < w;x++,a++){
			nlf xf = x/(nlf)(w-1);
			nlf yf = (y)/(nlf)(h-1);
			vin->data[0] = xf;
			vin->data[1] = yf;
			forward(act,nn,vin);
			vec outp = outputlayer(act);
			uc r = outp->data[0] * 255;
			uc g = outp->data[1] * 255;
			uc b = outp->data[2] * 255;
			rgba col = colour(r, g,b,255);
			context->pixels[a] = colToPixel(col);
		}

	destroyActivations(act);
	destroyVec(vin);
}
void drawIMG(MEDIA_H_CONTEXT * context, img image){
	if(context->scrH >= image->h && context->scrW >= image->w){
		for(uint x = 0; x < image->w;x++)
			for(uint y = 0; y < image->h;y++){
				setPixelValue(context, image->data[x][y], x,y);
			}
	}
}

void manage_keys(MEDIA_H_CONTEXT * context){
	if(GetKey(context, SDL_SCANCODE_UP))
    {
        context->scale -= 1;
		if(context->scale <= 0)context->scale = 1;
    }
	if(GetKey(context, SDL_SCANCODE_DOWN))
    {
        context->scale += 1;
		if(context->scale > 30)context->scale = 30;
    }
	if(GetKey(context, SDL_SCANCODE_R))
    {
        randModel(mdl);
    }
	static uc pausetoggle = 1;
	if(GetKey(context, SDL_SCANCODE_SPACE))
    {
		if(pausetoggle){
        	PAUSE +=1;
			PAUSE = PAUSE%2;
			pausetoggle = 0;
			displayModel(mdl);
		}
    }else{
		pausetoggle = 1;
	}
}

void draw(MEDIA_H_CONTEXT * context){
    char * title = malloc (30);
    sprintf(title, "FPS : %.1f",1.0/context->elapsedTime);
    ChangeTitle(context,title);
    free(title);

	manage_keys(context);
	draw_out(context,mdl);
	//drawIMG(context, test);
	//usleep(10000);//if i dont put it the program crashes i dont know why maybe too much fps?
}

int main(int argc,char ** argv){
	const char * png_file;
    if(argc == 1)png_file = "tuxedo.png";
    if(argc == 2)png_file = argv[1];
	test = loadIMG(png_file);
	PAUSE = 1;
	SDL_Init(SDL_INIT_EVERYTHING);

    MEDIA_H_CONTEXT context = newcontext("FPS : ",1000, 1000 ,500 ,500, SDL_WINDOW_RESIZABLE);
	context.scale = 10;
	srand(time(NULL));

	descriptor arch = newDescriptor(5,2,20,10,10,3);
	mdl = newModel(arch);
	model grad = newModel(arch);
	randModel(mdl);
	nlu w = test->w;
	nlu h = test->h;
	data_t data = newdataset(w*h, 2,3);
	nlu a = 0;
	for(nlu i = 0; i < h;i++){
		for(nlu j = 0; j < w;j++,a++){
			nlf x = j/(nlf)(w-1);
			nlf y = i/(nlf)(h-1);
			data.inputs[a][0] = x;
			data.inputs[a][1] = y;
			col c = test->data[j][i];
			col r = c >> 24;
			col g = c-(r<<24) >> 16;
			col b = c-((r<<24) + (g << 16)) >> 8;
			data.outputs[a][0] = r/(nlf)255;
			data.outputs[a][1] = g/(nlf)255;
			data.outputs[a][2] = b/(nlf)255;

		}
	}

	nlf rate = 1.0f;
	nlf c = cost(mdl,data);
	nlu i = 0;
	while(context.RUNNING){
		if(!PAUSE)stochastic_batch_descent(mdl,grad, data,test->w/2,rate);
		
		if(!PAUSE){
			if(i%100 == 0){
	        	update_MEDIA_H_CONTEXT(&context, draw);
			}
		}else{
			update_MEDIA_H_CONTEXT(&context, draw);
		}
		i++;
	}

	destroydataset(data);
	destroyModel(mdl);
	destroyModel(grad);
	destroyDesc(arch);
	destroycontext(context);
    SDL_Quit();
	destroyIMG(test);

  	return EXIT_SUCCESS;
}
