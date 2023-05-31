#ifndef MODEL_H
#define MODEL_H

typedef unsigned int u32;
typedef float f32;
typedef struct neuron_s{
	//weight count
	u32 wc;		
	//weight list
	f32 *w;	
	//bias
	f32 b;
    //activation
    f32 a;
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
//layer compute(model m);

#endif