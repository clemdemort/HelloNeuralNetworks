#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "src/matrix.h"
#include "src/model.h"


//currently Not working, bug hunting until i find the issue(s) things are looking up though :)


void _resetcol(){
    printf("\033[0m");                      //ANSI colour code
}

void gotoxy(int x,int y)
{
    printf("%c[%d;%df",0x1B,y,x);
}

void _setcol(uc r,uc g,uc b){
    printf("\033[38;2;%u;%u;%um",r,g,b);    //ANSI colour code
}

f32 distxy(f32 x, f32 y , f32 cx, f32 cy){
	return sqrt(((cx-x)*(cx-x)) + ((cy-y)*(cy-y)) );
}

void IMGvisualization(model nn,u32 w,u32 h){
	mat out = newMat(w,h);
	vec vin = newVec(2);
	descriptor D = getDescriptor(nn);
	activations act = newActivations(D);
	destroyDesc(D);
	for(u32 i = 0; i < h;i++){
		for(u32 j = 0; j < w;j++){
			f32 x = j/(f32)(w-1);
			f32 y = i/(f32)(h-1);
			vin->data[0] = x;
			vin->data[1] = y;
			forward(act,nn,vin);
			out->data[j][i] = outputlayer(act)->data[0];

		}
	}
	destroyActivations(act);
	destroyVec(vin);
	displayMatCol(out);
	destroyMat(out);
}

void IMGdata(data_t d,u32 w,u32 h){
	mat out = newMat(w,h);
	u32 a = 0;
	for(u32 i = 0; i < h;i++){
		for(u32 j = 0; j < w;j++,a++){
			out->data[j][i] = d.outputs[a][0];

		}
	}
	displayMatCol(out);
	destroyMat(out);
}

int main(){
	
	srand(time(NULL));
	descriptor arch = newDescriptor(4,2,15,5,1);
	model nn = newModel(arch);
	model grad = newModel(arch);
	randModel(nn);
	u32 w = 10;
	u32 h = 10;
	data_t data = newdataset(w*h, 2,1);
	u32 a = 0;
	for(u32 i = 0; i < h;i++){
		for(u32 j = 0; j < w;j++,a++){
			f32 x = j/(f32)(w-1);
			f32 y = i/(f32)(h-1);
			data.inputs[a][0] = x;
			data.inputs[a][1] = y;
			f32 d = distxy(0.5,0.5, x, y);
			data.outputs[a][0] = (d > 0.2 && d < 0.4);
			data.outputs[a][1] = (1);


		}
	}
	//*/

	f32 rate = 0.1f;
	system("clear");
	//sleep(1);
	for(u32 i = 0; i < 100000;i++){
		backpropagation(grad,nn,data);
		learn(nn,grad,rate);
		if(i%100 == 1){
			//sleep(1);
			gotoxy(0,0);
			IMGvisualization(nn,w,h);
			IMGdata(data, w, h);
			f32 c = cost(nn,data);
			printf("cost : %f\n",c);
			
		}
	}
	activations act = newActivations(arch);
	vec vinput = malloc(sizeof(vec_t));	  			//creating the input vector
	vinput->h = data.input_length;						//
	vinput->data = data.inputs[0];
	forward(act, nn, vinput);
	free(vinput);
	destroyActivations(act);
 	//displayModel(nn);

	destroydataset(data);
	destroyModel(nn);
	destroyModel(grad);
	destroyDesc(arch);
  	return EXIT_SUCCESS;
}
