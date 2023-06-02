#include <stdio.h>
#include <stdlib.h>
#include "model.h"
#include "matrix.h"

f32 sig(f32 x){
	return 1.f/(1.f + expf(-x));	
}


f32 randf32(){
	return (f32)rand()/(f32)RAND_MAX;
}



//wc : weight count -> how many weights should each neuron have (e.g how many neurons/entries before)
//nc : neuron count -> how many neurons in the layer
layer newlayer(u32 wc,u32 nc){
	layer res;
	res.nc = nc;
	res.n = malloc(sizeof(neuron_t)* nc);
	for(u32 i = 0; i < nc; i++ ){
		res.n[i].wc = wc;
		res.n[i].w = calloc(sizeof(f32) , wc);
		res.n[i].b = 0;
		res.n[i].a = 0;
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
	model res = malloc(sizeof(model_t));
	res->lc = arch.descsize;
	res->l = malloc(sizeof(layer)*res->lc);
	for(u32 i = 0; i < res->lc;i++){
		//i=0 will be the entry layer and i = res.lc-1 will be the exit layer
		//the rest (in between) will be hidden layers
		u32 prevnc = 0;
		if(i > 0)prevnc = arch.desc[i-1];
		res->l[i] = newlayer(prevnc, arch.desc[i]);

	}
	return res;
}

void destroyModel(model m){
	for(u32 i = 0; i < m->lc; i++){
		destroyLayer(m->l[i]);
	}
	free(m->l);
    free(m);
}

void compute(model m){
	for(u32 i = 1;i < m->lc;i++){
		mat m1 = weightstomat(m->l[i]);
		vec va = layertovec(m->l[i-1]);
		vec vb = biastovec(m->l[i]);

		//we do the math
		vec v1 = MatrixVectorProduct(m1, va);
		vec v2 = Vadd(v1,vb);
		forallVecElements(v2,sig);

		//we enter the results in the network;
		for(u32 j = 0; j < v2->h;j++){
			m->l[i].n[j].a = v2->data[j];
		}	

		//free the memory!!!
		destroyVec(va);
		destroyVec(vb);
		destroyVec(v1);
		destroyVec(v2);
		destroyMat(m1);
	}
}