#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "model.h"

f32 sig(f32 x){
	return 1.f/(1.f + expf(-x));	
}

f32 max(f32 a,f32 b){
	if(a > b)return a;
	return b;
}

f32 reLU(f32 x){
	return max(0,x);	
}


f32 randf32(){
	return (f32)rand()/(f32)RAND_MAX;
}



//wc : weight count -> how many weights should each neuron have (e.g how many neurons/entries before)
//nc : neuron count -> how many neurons in the layer
layer newlayer(u32 wc,u32 nc){
	layer res;

	res.weights = newMat(wc, nc);
	res.biases = newVec(nc);
	return res;
}

void destroyLayer(layer l){
	destroyVec(l.biases);
	destroyMat(l.weights);
}


model newModel(descriptor arch){
	model res = malloc(sizeof(model_t));
	res->lc = arch.descsize -1;	//we dont store the input layer
	res->l = malloc(sizeof(layer)*res->lc);
	for(u32 i = 0; i < res->lc;i++){
		//i = 0 means the first hidden layer
		res->l[i] = newlayer(arch.desc[i], arch.desc[i+1]);
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

vec forward(model m,vec vinput){
	vec vprev = vcpy(vinput);
	for(u32 i = 0;i < m->lc;i++){
		mat m1 = m->l[i].weights;
		vec va = vprev;
		vec vb = m->l[i].biases;

		//we do the math
		vec v1 = MatrixVectorProduct(m1, va);
		vec v2 = Vadd(v1,vb);
		forallVecElements(v2,sig);

		//we save the results in a vector
		vprev = v2;

		//free the memory!!!
		destroyVec(va);
		destroyVec(v1);
	}
	return vprev;
}

data_t newdataset(u32 entries,u32 inputs, u32 outputs){
	data_t res;
	res.entry_count = entries;
	res.input_length = inputs;
	res.output_length = outputs;
	res.inputs  = malloc(sizeof(f32*)*entries);
	res.outputs = malloc(sizeof(f32*)*entries);

	for(u32 i = 0; i < entries; i++){
		res.inputs[i]  = malloc(sizeof(f32)*inputs);
		res.outputs[i] = malloc(sizeof(f32)*outputs);
	}
	return res;
}

void freedataset(data_t data){
	for(u32 i = 0; i < data.entry_count; i++){
		free(data.inputs[i] );
		free(data.outputs[i]);
	}
	free(data.inputs);
	free(data.outputs);
}