#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "src/neuralLib.h"


void gotoxy(int x,int y)
{
    printf("%c[%d;%df",0x1B,y,x);
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
			nlu x = d.inputs[a][0]*(w-1);
			nlu y = d.inputs[a][1]*(h-1);
			out->data[x][y] = d.outputs[a][0];

		}
	}
	displayMatCol(out);
	destroyMat(out);
}

int main(){
	
	srand(time(NULL));

	descriptor arch = newDescriptor(4,2,12,7,1);
	model nn = newModel(arch);
	model grad = newModel(arch);
	randModel(nn);
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
	system("clear");
	for(nlu i = 0; i <= 30000;i++){
		stochastic_batch_descent(nn,grad, data,10,rate);

		if(i%100 == 0){
			gotoxy(0,0);
			IMGvisualization(nn,2*w,2*h);
			IMGdata(data, w,h);
			nlf c = cost(nn,data);
			printf("%u %f\n",i,c);
			if(c < 0.05) rate = 0.4;
			if(c > 0.05) rate = 1.0;
		}
	}

	destroydataset(data);
	destroyModel(nn);
	destroyModel(grad);
	destroyDesc(arch);
  	return EXIT_SUCCESS;
}
