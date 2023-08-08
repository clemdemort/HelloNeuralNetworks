#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "src/neuralLib.h"


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

nlf distxy(nlf x, nlf y , nlf cx, nlf cy){
	return sqrt(((cx-x)*(cx-x)) + ((cy-y)*(cy-y)) );
}

void IMGvisualization(model nn,nlu w,nlu h){
	mat out = newMat(w,h);
	vec vin = newVec(2);
	descriptor D = getDescriptor(nn);
	activations act = newActivations(D);
	destroyDesc(D);
	for(nlu i = 0; i < h;i++){
		for(nlu j = 0; j < w;j++){
			nlf x = j/(nlf)(w-1);
			nlf y = i/(nlf)(h-1);
			vin->data[0] = x;
			vin->data[1] = y;
			forward(act,nn,vin);
			vec outp = outputlayer(act);
			out->data[j][i] = outp->data[0];

		}
	}
	destroyActivations(act);
	destroyVec(vin);
	displayMatCol(out);
	destroyMat(out);
}

void IMGdata(data_t d,nlu w,nlu h){
	mat out = newMat(w,h);
	nlu a = 0;
	for(nlu i = 0; i < h;i++){
		for(nlu j = 0; j < w;j++,a++){
			out->data[j][i] = d.outputs[a][0];

		}
	}
	displayMatCol(out);
	destroyMat(out);
}

int main(){
	
	srand(time(NULL));
	//srand(69);
	descriptor arch = newDescriptor(4,2,15,5,1);
	model nn = newModel(arch);
	model grad = newModel(arch);
	randModel(nn);
	nlu w = 10;
	nlu h = 10;
	data_t data = newdataset(w*h, 2,1);
	nlu a = 0;
	for(nlu i = 0; i < h;i++){
		for(nlu j = 0; j < w;j++,a++){
			nlf x = j/(nlf)(w-1);
			nlf y = i/(nlf)(h-1);
			data.inputs[a][0] = x;
			data.inputs[a][1] = y;
			nlf d = distxy(0.5,0.5, x, y);
			data.outputs[a][0] = (d > 0.2 && d < 0.4);

		}
	}
	//*/

	nlf rate = 1.0f;
	system("clear");
	//sleep(1);
	for(nlu i = 0; i < 100000;i++){
		backpropagation(grad,nn,data);
		learn(nn,grad,rate);
		if(i%100 == 0){
			//sleep(1);
			gotoxy(0,0);
			IMGvisualization(nn,5*w,5*h);
			IMGdata(data, w, h);
			//visualization(nn, data);
			//displayModel(nn);
			nlf c = cost(nn,data);
			printf("%u %f\n",i,c);
			
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
