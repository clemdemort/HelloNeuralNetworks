#include "src/media.h"
#include "src/neuralLib.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

model mdl;//model

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
			nlf yf = y/(nlf)(h-1);
			vin->data[0] = xf;
			vin->data[1] = yf;
			forward(act,nn,vin);
			vec outp = outputlayer(act);
			uc c = outp->data[0] * 255;
			rgba col = colour(c, c,c,c);
			context->pixels[a] = colToPixel(col);
		}

	destroyActivations(act);
	destroyVec(vin);
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
}

void draw(MEDIA_H_CONTEXT * context){
    char * title = malloc (30);
    sprintf(title, "FPS : %.1f",1.0/context->elapsedTime);
    ChangeTitle(context,title);
    free(title);

	manage_keys(context);
    //RenderClear(context,colour(20,20,20,0));
	draw_out(context,mdl);
}


int main(){
	
	SDL_Init(SDL_INIT_EVERYTHING);

    MEDIA_H_CONTEXT context = newcontext("FPS : ",1000, 1000 ,500 ,500, SDL_WINDOW_RESIZABLE);
	context.scale = 4;
	srand(time(NULL));

	descriptor arch = newDescriptor(4,2,12,7,1);
	mdl = newModel(arch);
	model grad = newModel(arch);
	randModel(mdl);
	nlu w = 20;
	nlu h = 20;
	data_t data = newdataset(w*h, 2,1);
	nlu a = 0;
	for(nlu i = 0; i < h;i++){
		for(nlu j = 0; j < w;j++,a++){
			nlf x = j/(nlf)(w-1);
			nlf y = i/(nlf)(h-1);
			data.inputs[a][0] = x;
			data.inputs[a][1] = y;
			nlf d = distxy(0.5,0.5, x, y);
			data.outputs[a][0] = ((d > 0.3 && d < 0.5) || d < 0.2);

		}
	}

	nlf rate = 1.0f;
	nlf c = cost(mdl,data);
	nlu i = 0;
	while(context.RUNNING){
		stochastic_batch_descent(mdl,grad, data,10,rate);
		if(c < 0.05) rate = 0.4;
		if(c > 0.05) rate = 1.0;

		if(i%100 == 0){
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

  	return EXIT_SUCCESS;
}
