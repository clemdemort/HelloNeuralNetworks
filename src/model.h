#ifndef MODEL_H
#define MODEL_H

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


layer newlayer(u32 wc,u32 nc);
void destroyLayer(layer l);
model newModel(descriptor arch);
void destroyModel(model m);

#endif