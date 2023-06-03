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


/*
f32 cost(neuron n){
	f32 res = 0.0f;
	for(size_t i = 0; i < data_count; i++){
		f32 y = 0;
		for(u32 j = 0; j < n->wc;j++){
			f32 x = data[i][j];
			y += x * n->w[j];
		}
		y = sig(y + n->b);
		f32 d = y - data[i][2];
		res += d*d;

	}
	res /= data_count;
	return res;
}
void train(neuron n,f32 eps, f32 rate){
	f32 * dw = malloc(n->wc*sizeof(f32));
	f32 c = cost(n);
	for(u32 j = 0; j < n->wc;j++){
		f32 save = n->w[j];
		n->w[j] += eps;
		dw[j] = rate * ((cost(n)-c)/eps);
		n->w[j] = save;
	}
	f32 save  = n->b;
	n->b += eps;
	f32 db = rate * ((cost(n)-c)/eps);
	n->b = save;
	n->b -= db;
	for(u32 j = 0; j < n->wc;j++){
		n->w[j] -= dw[j];
	}
	free(dw);
	//printf("cost : %f \tW1 : %f \tW2 : %f \tB: %f \n",cost(n),n->w[0],n->w[1],n->b);
	//printf("%f\n",cost(w1,w2,b));
}*/
int main(){
	/*
	srand(time(0));
	neuron n = newNeuron(2);
	//printf("w1 : %f, w2 : %f, B : %f , cost : %f\n",w1,w2,b,c);
	f32 eps = 1e-1;
	f32 rate = 1e-1;

	for(size_t i = 0; i < data_count; i++){
		f32 x1 = data[i][0];
		f32 x2 = data[i][1];
		f32 y = sig((x1*n->w[0]) + (x2*n->w[1]) + n->b);
		printf("actual : %f \texpected : %f\n",y,data[i][2]);

	}
	printf("cost : %f\n",cost(n));

	u32 tc = 10000000;
	printf("training %u times\n",tc);
	for(u32 i = 0; i < tc; i++){
		train(n, eps, rate);
	}
	
	for(size_t i = 0; i < data_count; i++){
		f32 x1 = data[i][0];
		f32 x2 = data[i][1];
		f32 y = sig((x1*n->w[0]) + (x2*n->w[1]) + n->b);
		printf("actual : %f \texpected : %f\n",y,data[i][2]);

	}
	printf("cost : %f\n",cost(n));*/
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

	vec res = compute(nn,in);	//magic!
	printf("input : \n");
	displayVec(in);
	printf("result : \n");
	displayVec(res);


	destroyVec(in);
	destroyVec(res);
	destroyModel(nn);
	free(arch.desc);
  	return EXIT_SUCCESS;
}
