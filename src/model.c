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

//creates a descriptor to create the neural network
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

activations forward(model m,vec vinput){
	vec vprev = vcpy(vinput);
	activations a = malloc(sizeof(activations_t));
	a->lc = m->lc+1;
	a->layers = malloc(sizeof(vec_t)*a->lc);
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
		destroyVec(v1);
		a->layers[i] = *va;
		free(va);	//needs to be freed because it was malloc'd into existence
	}
	a->layers[a->lc-1] = *vprev;
	free(vprev);	//needs to be freed because it was malloc'd into existence
	return a;
}

void destroyActivations(activations a){
	for(u32 i = 0; i < a->lc; i++){
		free(a->layers[i].data);
	}
	free(a->layers);
	free(a);
}

vec outputlayer(activations a){
	return &a->layers[a->lc-1];
}

void displayActivations(activations a){
	for(u32 i = 0; i < a->lc; i++){
		printf("layer %u :\n",i);
		for(u32 j = 0; j < a->layers[i].h; j++){
			printf(" %f,",a->layers[i].data[j]);
		}
		printf("\n");
	}
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

void HumanVerification(model nn,data_t data){
	for(size_t i = 0; i < data.entry_count; i++){		 	//for all entries
		vec vinput = malloc(sizeof(vec_t));	  	//creating the input vector
		vinput->h = data.input_length;						//
		vinput->data = data.inputs[i];		 				//putting the data inside

		activations resA = forward(nn, vinput);				 //forwarding the model with the input vector
		vec resc = outputlayer(resA);
		for(u32 i = 0; i < vinput->h;i++){
			printf(" %f",vinput->data[i]);
		}
		printf(" =");
		for(u32 k = 0; k < data.output_length;k++){
			printf(" %f",resc->data[k]);		
		}
		printf("\n");
		//cleaning
		free(vinput);
		destroyActivations(resA);

	}
}

f32 cost(model m,data_t e){
	f32 res = 0.0f;
	for(size_t i = 0; i < e.entry_count; i++){		 	//for all entries
		vec vinput = malloc(sizeof(vec_t));	  	//creating the input vector
		vinput->h = e.input_length;						//
		vinput->data = e.inputs[i];		 				//putting the data inside

		activations resA = forward(m, vinput);				 //forwarding the model with the input vector
		vec resc = outputlayer(resA);
		for(u32 k = 0; k < e.output_length;k++){
			//printf("resc->data[%u] = %f e.outputs[%u][%u] = %f\n",k,resc->data[k],i,k,e.outputs[i][k]);
			f32 d = resc->data[k] - e.outputs[i][k]; //calculating the difference			
			res += d*d;								 //squaring the total (IDK why but 3b1b said so) and adding it up
		}
		//cleaning
		free(vinput);
		destroyActivations(resA);

	}
	//getting the mean
	res /= (float)e.entry_count;
	return res;
}

//wip

model nn_backpropagation(model nn,data_t e){
	if(nn->l[0].weights->w != e.input_length || nn->l[nn->lc-1].biases->h != e.output_length){printf("input/output mismatch between model and dataset\n");return NULL;}
	descriptor arch = getDescriptor(nn);
    model g = newModel(arch);	//G for gradient

	for(u32 i = 0; i < e.entry_count;i++){
		vec vinput = malloc(sizeof(vec_t));	  	//creating the input vector
		vinput->h = e.input_length;						//
		vinput->data = e.inputs[i];		 				//putting the data inside
		activations act = forward(nn, vinput);
		free(vinput);
		destroyActivations(act);
	}
	destroyDesc(arch);
	return g;
}

//inefficient but works
model finite_diff(model nn, data_t t, float eps)
{
    f32 saved;
    f32 c = cost(nn, t);

	descriptor arch = getDescriptor(nn);
    model g = newModel(arch);
	destroyDesc(arch);

	for(u32 i = 0; i < nn->lc;i++){//for every layer
		for(u32 j = 0; j < nn->l[i].weights->h;j++){//for every activation
			for(u32 k = 0; k < nn->l[i].weights->w;k++){//for every weight
				saved = nn->l[i].weights->data[k][j];
				nn->l[i].weights->data[k][j] += eps;
				g->l[i].weights->data[k][j] = (cost(nn,t)-c)/eps;
				nn->l[i].weights->data[k][j] = saved;
			}

			saved = nn->l[i].biases->data[j];
			nn->l[i].biases->data[j] += eps;
			g->l[i].biases->data[j] = (cost(nn,t)-c)/eps;
			nn->l[i].biases->data[j] = saved;

		}
	}

    return g;
}

void learn(model nn, model g, float rate){

	if(nn->lc != g->lc)printf("model and gradient dont have the same architecture\n");
	if(nn->l[0].biases->h != g->l[0].biases->h)printf("model and gradient dont have the same architecture\n");

	for(u32 i = 0; i < nn->lc;i++){//for every layer
		for(u32 j = 0; j < nn->l[i].weights->h;j++){//for every activation
			for(u32 k = 0; k < nn->l[i].weights->w;k++){//for every weight
				nn->l[i].weights->data[k][j] -= g->l[i].weights->data[k][j]*rate;
			}
			nn->l[i].biases->data[j] -= g->l[i].biases->data[j]*rate;

		}
	}
}

void displayModel(model nn){
	for(u32 i = 0; i < nn->lc;i++){
		printf("layer %u:\n",i);
		displayMat(nn->l[i].weights);
		displayVec(nn->l[i].biases);
	}
}
