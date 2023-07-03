#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "model.h"

//sigmoid function
f32 sig(f32 x){
	return 1.f/(1.f + expf(-x));	
}

//derivative of sigmoid function
f32 Dsig(f32 x){
	return sig(x)*(1-sig(x));
}

f32 max(f32 a,f32 b){
	if(a > b)return a;
	return b;
}

f32 reLU(f32 x){
	return max(0,x);	
}

//creates a descriptor to creae the neural network
//the first argument is the numbers of layers
//each argument you put next is the amount of neurons inside the layer
//make sure that if you specify 3 layers you add 3 arguments otherwise weird things might happen.
descriptor newDescriptor(u32 descSize, ...){
	
	descriptor arch;
	arch.descsize = descSize;
	arch.desc = malloc(sizeof(u32)*arch.descsize);

	// Declaring pointer to the
    // argument list
    va_list ptr;
 
    // Initializing argument to the
    // list pointer
    va_start(ptr, descSize);
	
 
    for (u32 i = 0; i < descSize; i++) {
 
        u32 val = va_arg(ptr, u32);
		if(val > 10000){printf("[newDescriptor WARNING] %u Layer is huge (%u neurons) are you sure you didn't misuse the function?\n ",i,val);}
		arch.desc[i] = val;
    }
 
    // End of argument list traversal
    va_end(ptr);
 
    return arch;
}

descriptor getDescriptor(model nn){
	descriptor arch;
	arch.descsize = nn->lc+1;
	arch.desc = malloc(arch.descsize);
	arch.desc[0] = nn->l[0].weights->w;
	for(u32 i = 1; i < arch.descsize;i++){
		arch.desc[i] = nn->l[i-1].biases->h;
	}
	return arch;
}

void destroyDesc(descriptor arch){
	free(arch.desc);
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

void zeroModel(model m){
	for(u32 i = 0; i < m->lc; i++){
		zeroVec(m->l[i].biases);
		zeroMat(m->l[i].weights);
	}
}

void randModel(model m){
	for(u32 i = 0; i < m->lc; i++){
		randVec(m->l[i].biases);
		randMat(m->l[i].weights);
	}
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

void destroydataset(data_t data){
	for(u32 i = 0; i < data.entry_count; i++){
		free(data.inputs[i] );
		free(data.outputs[i]);
	}
	free(data.inputs);
	free(data.outputs);
}

f32 cost(model m,data_t e){
	f32 res = 0.0f;
	for(size_t i = 0; i < e.entry_count; i++){		 //for all entries
		vec vinput = newVec(e.input_length); //creating the input vector
		for(u32 j = 0; j < e.input_length;j++){
			vinput->data[j] = e.inputs[i][j];		 //putting the data inside
		}

		vec resc = forward(m, vinput);				 //forwarding the model with the input vector
		for(u32 k = 0; k < e.output_length;k++){
			f32 d = resc->data[k] - e.outputs[i][k]; //calculating the difference			
			res += d*d;								 //squaring the total (IDK why but 3b1b said so) and adding it up
		}
		//cleaning
		destroyVec(vinput);
		destroyVec(resc);

	}
	//getting the mean
	res /= (float)e.entry_count;
	return res;
}

