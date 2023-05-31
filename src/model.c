#include <stdio.h>
#include <stdlib.h>
#include "model.h"

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