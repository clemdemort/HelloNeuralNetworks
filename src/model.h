#ifndef MODEL_H
#define MODEL_H
#include "matrix.h"
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

typedef struct data_s{
	u32 entry_count;
	u32 input_length;
	u32 output_length;
	f32 ** inputs;
	f32 ** outputs;
}data_t;

typedef model_t * model;

typedef struct descriptor_s{
	u32 * desc;
	u32 descsize;
}descriptor;

layer newlayer(u32 wc,u32 nc);
void destroyLayer(layer l);
model newModel(descriptor arch);
void destroyModel(model m);
void zeroModel(model m);
void randModel(model m);
//forward : a function that takes in a model and modifies it's output layer
vec forward(model m,vec vinput);
f32 sig(f32 x);
f32 reLU(f32 x);
data_t newdataset(u32 entries,u32 inputs, u32 outputs);
void destroydataset(data_t data);
//cost function takes in as input a model and a dataset and evaluates how close does the model
//comes to replicating the dataset
f32 cost(model m,data_t e);


#endif