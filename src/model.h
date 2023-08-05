#ifndef MODEL_H
#define MODEL_H
#include "matrix.h"
#include <stdarg.h>
#include <sys/types.h>
typedef unsigned int u32;
typedef float f32;

typedef struct layer_s{
	mat weights;
	vec biases;
}layer;

typedef struct model_s{
	//layer count	
	u32 lc;			
	//layer list
	layer * l;	
}model_t; 
typedef model_t * model;

typedef struct activations_s{
	//layer count
	u32 lc;
	//activations stored in layers
	vec_t * layers;
}activations_t;

typedef activations_t * activations;


typedef struct data_s{
	u32 entry_count;
	u32 input_length;
	u32 output_length;
	f32 ** inputs;
	f32 ** outputs;
}data_t;


typedef struct descriptor_s{
	u32 * desc;
	u32 descsize;
}descriptor;

//allocates memory
descriptor newDescriptor(u32 descSize, ...);
//allocates memory
descriptor getDescriptor(model nn);

void destroyDesc(descriptor arch);


layer newlayer(u32 wc,u32 nc);
void destroyLayer(layer l);
model newModel(descriptor arch);
void destroyModel(model m);
void zeroModel(model m);
void randModel(model m);


//forward : a function that takes in a model and feeds it input data then computes the activations from that
void forward(activations a, model m,vec vinput);
activations newActivations(descriptor D);
void destroyActivations(activations A);
void displayActivations(activations a);
void zeroActivations(activations a);
//does not allocate anything it just looks something up
vec outputlayer(activations a);
f32 sig(f32 x);
f32 reLU(f32 x);
data_t newdataset(u32 entries,u32 inputs, u32 outputs);
void destroydataset(data_t data);
//cost function takes in as input a model and a dataset and evaluates how close does the model
//comes to replicating the dataset
f32 cost(model m,data_t e);
void displayModel(model nn);
void finite_diff(model g,model nn, data_t t, f32 eps);
void backpropagation(model g, model nn,data_t e);
void learn(model nn, model g, f32 rate);
void HumanVerification(model nn,data_t data);
void visualization(model nn,data_t data);

#endif