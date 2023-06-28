#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "src/matrix.h"
#include "src/model.h"

typedef f32 cpl[3];
cpl data[] = {
	{0,0,0},
	{0,1,1},
	{1,0,1},
	{1,1,0},
};
#define data_count (sizeof(data)/sizeof(data[0]))


//need to make a "global" cost function so that i can input whatever i want into it and not
//have to change it, this will come in handy for backpropagation.
f32 cost(model m,data_t e){
	f32 res = 0.0f;
	for(size_t i = 0; i < e.entry_count; i++){
		vec vinput = newVec(2);
		for(u32 j = 0; j < e.input_length;j++){
			vinput->data[j] = e.inputs[i][j];
		}

		vec resc = forward(m, vinput);
		for(u32 k = 0; k < e.output_length;k++){
			f32 y = resc->data[k];
			f32 d = y - e.outputs[i][k];
			res += d*d;
		}
		destroyVec(vinput);
		destroyVec(resc);

	}
	res /= (float)e.entry_count;
	return res;
}

int main(){
	descriptor arch;
	arch.descsize = 3;
	arch.desc = malloc(sizeof(u32)*arch.descsize);
	arch.desc[0] = 2;	//entry layer
	arch.desc[1] = 2;	//hiden layer(s)
	arch.desc[2] = 1;	//exit  layer
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
	vec res = forward(nn,in);	//magic!
	printf("input : \n");
	displayVec(in);
	printf("result : \n");
	displayVec(res);
	printf("cost : %f\n",cost(nn,data));

	destroyVec(in);
	destroyVec(res);
	destroyModel(nn);
	free(arch.desc);
  	return EXIT_SUCCESS;
}
