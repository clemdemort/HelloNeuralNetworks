#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "src/matrix.h"


typedef unsigned int u32;
typedef struct neuron_s{
	u32 wc;		//weight count
	float *w;	//weights
	float b;	//bias
}neuron_t;
typedef neuron_t * neuron;

neuron newNeuron(u32 wc){
	neuron res = malloc(sizeof(neuron_t));
	res->wc = wc;
	res->w = calloc(sizeof(float) , wc);
	return res;
}

void destroyNeuron(neuron n){
	free(n->w);
	free(n);
}

typedef float cpl[3];
cpl data[] = {
	{0,0,0},
	{0,1,1},
	{1,0,1},
	{1,1,1},
};
#define data_count (sizeof(data)/sizeof(data[0]))


float sig(float x){
	return 1.f/(1.f + expf(-x));	
}


float randfloat(){
	return (float)rand()/(float)RAND_MAX;
}

float cost(neuron n){
	float res = 0.0f;
	for(size_t i = 0; i < data_count; i++){
		float y = 0;
		for(u32 j = 0; j < n->wc;j++){
			float x = data[i][j];
			y += x * n->w[j];
		}
		y = sig(y + n->b);
		float d = y - data[i][2];
		res += d*d;

	}
	res /= data_count;
	return res;
}
void train(neuron n,float eps, float rate){
	float * dw = malloc(n->wc*sizeof(float));
	float c = cost(n);
	for(u32 j = 0; j < n->wc;j++){
		n->w[j] += eps;
		dw[j] = rate * ((cost(n)-c)/eps);
		n->w[j] -= eps;
	}
	n->b += eps;
	float db = rate * ((cost(n)-c)/eps);
	n->b -= eps;
	n->b -= db;
	for(u32 j = 0; j < n->wc;j++){
		n->w[j] -= dw[j];
	}
	free(dw);
	//printf("cost : %f \tW1 : %f \tW2 : %f \tB: %f \n",cost(n),n->w[0],n->w[1],n->b);
	//printf("%f\n",cost(w1,w2,b));
}
int main(){
	srand(time(0));
	neuron n = newNeuron(2);
	//printf("w1 : %f, w2 : %f, B : %f , cost : %f\n",w1,w2,b,c);
	float eps = 1e-1;
	float rate = 1e-1;

	for(size_t i = 0; i < data_count; i++){
		float x1 = data[i][0];
		float x2 = data[i][1];
		float y = sig((x1*n->w[0]) + (x2*n->w[1]) + n->b);
		printf("actual : %f \texpected : %f\n",y,data[i][2]);

	}
	printf("cost : %f\n",cost(n));

	u32 tc = 10000000;
	printf("training %u times\n",tc);
	for(u32 i = 0; i < tc; i++){
		train(n, eps, rate);
	}
	
	for(size_t i = 0; i < data_count; i++){
		float x1 = data[i][0];
		float x2 = data[i][1];
		float y = sig((x1*n->w[0]) + (x2*n->w[1]) + n->b);
		printf("actual : %f \texpected : %f\n",y,data[i][2]);

	}
	printf("cost : %f\n",cost(n));

  	return EXIT_SUCCESS;
}
