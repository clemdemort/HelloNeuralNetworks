#ifndef MODEL_H
#define MODEL_H
#include "matrix.h"
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

typedef struct descriptor_s{
	u32 * desc;
	u32 descsize;
}descriptor;


layer newlayer(u32 wc,u32 nc);
void destroyLayer(layer l);
model newModel(descriptor arch);
void destroyModel(model m);
//compute : a function that takes in a model and modifies it's output layer
vec compute(model m,vec vinput);
f32 sig(f32 x);
f32 reLU(f32 x);


#endif