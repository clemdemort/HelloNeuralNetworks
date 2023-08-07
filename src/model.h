#ifndef MODEL_H
#define MODEL_H
#include "matrix.h"
#include <stdarg.h>
#include <sys/types.h>
typedef unsigned int nlu;
typedef float nlf;

typedef struct layer_s{
	mat weights;
	vec biases;
}layer;

typedef struct model_s{
	//layer count	
	nlu lc;			
	//layer list
	layer * l;	
}model_t; 
typedef model_t * model;

typedef struct activations_s{
	//layer count
	nlu lc;
	//activations stored in layers
	vec_t * layers;
}activations_t;

typedef activations_t * activations;


typedef struct data_s{
	nlu entry_count;
	nlu input_length;
	nlu output_length;
	nlf ** inputs;
	nlf ** outputs;
}data_t;


typedef struct descriptor_s{
	nlu * desc;
	nlu descsize;
}descriptor;

//allocates memory
descriptor newDescriptor(nlu descSize, ...);
//allocates memory
descriptor getDescriptor(model nn);

void destroyDesc(descriptor arch);


layer newlayer(nlu wc,nlu nc);
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
nlf sig(nlf x);
nlf reLU(nlf x);
data_t newdataset(nlu entries,nlu inputs, nlu outputs);
void destroydataset(data_t data);
//cost function takes in as input a model and a dataset and evaluates how close does the model
//comes to replicating the dataset
nlf cost(model m,data_t e);
void displayModel(model nn);
void finite_diff(model g,model nn, data_t t, nlf eps);
void backpropagation(model g, model nn,data_t e);
void learn(model nn, model g, nlf rate);
void HumanVerification(model nn,data_t data);
void visualization(model nn,data_t data);

#endif