#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "src/matrix.h"
#include "src/model.h"

int main(){
	srand(time(NULL));
	descriptor arch = newDescriptor(3, 2,2,1);
	model nn = newModel(arch);
	//manually training the model for fun
				//XOR gate
	//or
	nn->l[0].weights->data[0][0] = 10.0f;	
	nn->l[0].weights->data[1][0] = 10.0f;	
	nn->l[0].biases->data[0]	 = -5.0f;	
	
	//nand
	nn->l[0].weights->data[0][1] = -10.0f;	
	nn->l[0].weights->data[1][1] = -10.0f;	
	nn->l[0].biases->data[1]	 =  15.0f;	
	
	//and
	nn->l[1].weights->data[0][0] =  10.0f;	
	nn->l[1].weights->data[1][0] =  10.0f;	
	nn->l[1].biases->data[0]	 = -15.0f;		


	//input:
	//change these if you want to see if it works

	vec in = newVec(2);
	in->data[0] = 1.0f;
	in->data[1] = 0.0f;

	printf("setting up dataset\n");

	data_t data = newdataset(4, 2,1);
	data.inputs[0][0] = 0;
	data.inputs[0][1] = 0;
	data.inputs[1][0] = 1;
	data.inputs[1][1] = 0;
	data.inputs[2][0] = 0;
	data.inputs[2][1] = 1;
	data.inputs[3][0] = 1;
	data.inputs[3][1] = 1;

	data.outputs[0][0] = 0;
	data.outputs[1][0] = 1;
	data.outputs[2][0] = 1;
	data.outputs[3][0] = 0;


	printf("calc\n");
	//randModel(nn);	//if we want to randomize the model
	//zeroModel(nn);	//if we want every value in the model to be 0
	vec res = forward(nn,in);	//magic!
	printf("input : \n");
	displayVec(in);
	printf("result : \n");
	displayVec(res);
	printf("cost : %f\n",cost(nn,data));
	destroydataset(data);
	destroyVec(in);
	destroyVec(res);
	destroyModel(nn);
	destroyDesc(arch);
  	return EXIT_SUCCESS;
}
