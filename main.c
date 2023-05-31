#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "src/matrix.h"


typedef unsigned int u32;
typedef struct neuron_s{
	//weight count
	u32 wc;		
	//weight list
	float *w;	
	//bias
	float b;	
}neuron_t;

typedef struct layer_s{
	//neuron count	
	u32 nc;			
	//neuron list
	neuron_t * n;	
}layer;

typedef struct model_s{
	//layer count	
	u32 lc;			
	//layer list
	layer * l;	
}model;

typedef struct descriptor_s{
	u32 * desc;
	u32 descsize;
}descriptor;

//wc : weight count -> how many weights should each neuron have (e.g how many neurons/entries before)
//nc : neuron count -> how many neurons in the layer
layer newlayer(u32 wc,u32 nc){
	layer res;
	res.nc = nc;
	res.n = malloc(sizeof(neuron_t)* nc);
	for(u32 i = 0; i < nc; i++ ){
		res.n[i].wc = wc;
		res.n[i].w = calloc(sizeof(float) , wc);
		res.n[i].b = 0;
	}
	return res;
}

void destroyLayer(layer l){
	for(u32 i = 0; i < l.nc; i++ ){
		free(l.n[i].w);
	}
	free(l.n);
}


model newModel(descriptor arch){
	model res;
	res.lc = arch.descsize;
	res.l = malloc(sizeof(layer)*res.lc);
	for(u32 i = 0; i < res.lc;i++){
		//i=0 will be the entry layer and i = res.lc-1 will be the exit layer
		//the rest (in between) will be hidden layers
		u32 prevnc = 0;
		if(i > 0)prevnc = arch.desc[i-1];
		res.l[i] = newlayer(prevnc, arch.desc[i]);

	}
	return res;
}

void destroyModel(model m){
	for(u32 i = 0; i < m.lc; i++){
		destroyLayer(m.l[i]);
	}
	free(m.l);
}

typedef float cpl[3];
cpl data[] = {
	{0,0,0},
	{0,1,1},
	{1,0,1},
	{1,1,0},
};
#define data_count (sizeof(data)/sizeof(data[0]))


float sig(float x){
	return 1.f/(1.f + expf(-x));	
}


float randfloat(){
	return (float)rand()/(float)RAND_MAX;
}
/*
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
		float save = n->w[j];
		n->w[j] += eps;
		dw[j] = rate * ((cost(n)-c)/eps);
		n->w[j] = save;
	}
	float save  = n->b;
	n->b += eps;
	float db = rate * ((cost(n)-c)/eps);
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
	printf("cost : %f\n",cost(n));*/
	descriptor arch;
	arch.descsize = 3;
	arch.desc = malloc(sizeof(u32)*arch.descsize);
	arch.desc[0] = 2;
	arch.desc[1] = 2;
	arch.desc[2] = 1;
	model nn = newModel(arch);
	destroyModel(nn);
	free(arch.desc);
  	return EXIT_SUCCESS;
}
