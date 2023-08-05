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
	return sig(x)*(1.f-sig(x));
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
descriptor newDescriptor(u32 descSize, ...){//works
	
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

descriptor getDescriptor(model nn){//works
	descriptor arch;
	arch.descsize = nn->lc+1;
	arch.desc = malloc(arch.descsize*sizeof(u32));
	arch.desc[0] = nn->l[0].weights->w;
	for(u32 i = 1; i < arch.descsize;i++){
		arch.desc[i] = nn->l[i-1].biases->h;
	}
	return arch;
}

void destroyDesc(descriptor arch){//works
	free(arch.desc);
}


//wc : weight count -> how many weights should each neuron have (e.g how many neurons/entries before)
//nc : neuron count -> how many neurons in the layer
layer newlayer(u32 wc,u32 nc){//works
	layer res;

	res.weights = newMat(wc, nc);
	res.biases = newVec(nc);
	return res;
}

void destroyLayer(layer l){//works
	destroyVec(l.biases);
	destroyMat(l.weights);
}


model newModel(descriptor arch){//works
	model res = malloc(sizeof(model_t));
	res->lc = arch.descsize -1;	//we dont store the input layer
	res->l = malloc(sizeof(layer)*res->lc);
	for(u32 i = 0; i < res->lc;i++){
		//i = 0 means the first hidden layer
		res->l[i] = newlayer(arch.desc[i], arch.desc[i+1]);
	}
	return res;
}

void zeroModel(model m){//works
	for(u32 i = 0; i < m->lc; i++){
		zeroVec(m->l[i].biases);
		zeroMat(m->l[i].weights);
	}
}

void randModel(model m){//works
	for(u32 i = 0; i < m->lc; i++){
		randVec(m->l[i].biases);
		randMat(m->l[i].weights);
	}
}

void destroyModel(model m){//works
	for(u32 i = 0; i < m->lc; i++){
		destroyLayer(m->l[i]);
	}
	free(m->l);
    free(m);
}

void forward(activations a, model m,vec vinput){//probably works (hard to test)
	//SHOULD PUT VERIFICATION TO SEE IF ACTIVATION IS CORRECTLY ALLOCATED
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
		destroyVec(v1);
		free(a->layers[i].data);
		a->layers[i].data = va->data;
		free(va);	//needs to be freed because it was malloc'd into existence
	}
	//needless to say this is a dirty solution, however it is the fastest i can think of complexity wise
	free(a->layers[a->lc-1].data);		//freeing the preexisting data
	a->layers[a->lc-1].data = vprev->data;	//putting the new data in it's stead
	free(vprev);	//needs to be freed because it was malloc'd into existence
}

activations newActivations(descriptor D){//works
	activations a = malloc(sizeof(activations_t));
	a->lc = D.descsize;
	a->layers = malloc(sizeof(vec_t)*a->lc);
	for(u32 i = 0; i < D.descsize; i++){
		a->layers[i].h = D.desc[i];
		a->layers[i].data = calloc(D.desc[i],sizeof(f32));
	}
	return a;
}

void destroyActivations(activations a){//works
	for(u32 i = 0; i < a->lc; i++){
		free(a->layers[i].data);
	}

	free(a->layers);
	free(a);
}

void zeroActivations(activations a){//works
	for(u32 i = 0; i < a->lc; i++){
		for(u32 j = 0; j < a->layers[i].h;j++){
			a->layers[i].data[j] = 0;
		}
	}
}


vec outputlayer(activations a){//works
	return &a->layers[a->lc-1];
}

void displayActivations(activations a){//works
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
	descriptor D = getDescriptor(nn);
	activations resA = newActivations(D);
	destroyDesc(D);
	for(size_t i = 0; i < data.entry_count; i++){		 	//for all entries
		vec vinput = malloc(sizeof(vec_t));	  	//creating the input vector
		vinput->h = data.input_length;						//
		vinput->data = data.inputs[i];		 				//putting the data inside

		forward(resA , nn, vinput);				 //forwarding the model with the input vector
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

	}
	destroyActivations(resA);
}

void visualization(model nn,data_t data){
	descriptor D = getDescriptor(nn);
	activations resA = newActivations(D);
	destroyDesc(D);
	mat out = newMat(data.output_length,data.entry_count);
	for(size_t i = 0; i < data.entry_count; i++){		 	//for all entries
		vec vinput = malloc(sizeof(vec_t));	  	//creating the input vector
		vinput->h = data.input_length;						//
		vinput->data = data.inputs[i];		 				//putting the data inside

		forward(resA, nn, vinput);				 //forwarding the model with the input vector
		vec resc = outputlayer(resA);
		for(u32 k = 0; k < data.output_length;k++){
			out->data[k][i] = resc->data[k];		
		}
		//cleaning
		free(vinput);

	}
	displayMatCol(out);
	destroyMat(out);
	destroyActivations(resA);
}

f32 cost(model m,data_t e){
	f32 res = 0.0f;
	descriptor D = getDescriptor(m);
	activations resA = newActivations(D);
	destroyDesc(D);
	for(size_t i = 0; i < e.entry_count; i++){		 	//for all entries
		vec vinput = malloc(sizeof(vec_t));	  			//creating the input vector
		vinput->h = e.input_length;						//
		vinput->data = e.inputs[i];		 				//putting the data inside

		forward(resA, m, vinput);				 	//forwarding the model with the input vector
		vec resc = outputlayer(resA);
		for(u32 k = 0; k < e.output_length;k++){
			//printf("resc->data[%u] = %f e.outputs[%u][%u] = %f\n",k,resc->data[k],i,k,e.outputs[i][k]);
			f32 d = resc->data[k] - e.outputs[i][k]; //calculating the difference			
			res += d*d;								 //squaring the total (IDK why but 3b1b said so) and adding it up
		}
		//cleaning
		free(vinput);

	}
	//getting the mean
	res /= (f32)e.entry_count;
	destroyActivations(resA);
	return res;
}


//wip (testing)
void backpropagation(model g,model nn,data_t e){
	//SHOULD PUT VERIFICATION TO SEE IF ACTIVATION IS CORRECTLY ALLOCATED
	if(nn->l[0].weights->w != e.input_length || nn->l[nn->lc-1].biases->h != e.output_length){printf("input/output mismatch between model and dataset\n");}
	descriptor arch = getDescriptor(nn);
	activations GA = newActivations(arch);
	activations act = newActivations(arch);
	destroyDesc(arch);
	zeroModel(g);
	u32 n = e.entry_count;
	for(u32 i = 0; i < e.entry_count;i++){
		vec vinput = malloc(sizeof(vec_t));	  	//creating the input vector
		vinput->h = e.input_length;						//
		vinput->data = e.inputs[i];		 				//putting the data inside
		forward(act,nn, vinput);
		free(vinput);


		//create empty activations for G
		zeroActivations(GA);
		//calculate Dcost in output of G
		//this is the error
        for (u32 j = 0; j < outputlayer(act)->h; j++) {
			outputlayer(GA)->data[j] = 2*(outputlayer(act)->data[j] - e.outputs[i][j]);
			//printf("%f\n",outputlayer(GA)->data[j]);
		}

		for(u32 l = GA->lc-1; l > 0; l--){//for each layer starting by the end
			for(u32 j = 0; j < GA->layers[l].h;j++){//for each activation of that layer
				f32 a = act->layers[l].data[j];
                f32 da = GA->layers[l].data[j];
                f32 qa = Dsig(a);
				//printf("Layer = %u DA = %f\n",l,da);
				g->l[l-1].biases->data[j] += da*qa;
                for (size_t k = 0; k < GA->layers[l-1].h; ++k) {
					//j = height
					//k = width
                    f32 pa = act->layers[l-1].data[k];
                    f32 w = nn->l[l-1].weights->data[k][j];
					g->l[l-1].weights->data[k][j] += da*qa*pa;
					//printf("DA = %f\n",da);
                    GA->layers[l-1].data[k] += da*qa*w;
					//printf("QA = %f\n",qa);
                }
			}
		}
	}
	destroyActivations(act);

	//displayModel(g);
	for (size_t i = 0; i < g->lc-1; ++i) {
    	for (size_t j = 0; j < g->l[i].weights->h; ++j) {
	        for (size_t k = 0; k < g->l[i].weights->w; ++k) {
				g->l[i].weights->data[k][j] /= (f32)n;
            }
			g->l[i].biases->data[j] /= (f32)n;
        }
    }
	destroyActivations(GA);
}

//inefficient but works
void finite_diff(model g, model nn, data_t t, f32 eps)
{
	//SHOULD PUT VERIFICATION TO SEE IF ACTIVATION IS CORRECTLY ALLOCATED
    f32 saved;
    f32 c = cost(nn, t);
	zeroModel(g);
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

}

void learn(model nn, model g, f32 rate){

	//if(nn->lc != g->lc)printf("model and gradient dont have the same architecture\n");
	//if(nn->l[0].biases->h != g->l[0].biases->h)printf("model and gradient dont have the same architecture\n");

	for(u32 i = 0; i < nn->lc;i++){//for every layer
		for(u32 j = 0; j < nn->l[i].weights->h;j++){//for every activation
			for(u32 k = 0; k < nn->l[i].weights->w;k++){//for every weight
				nn->l[i].weights->data[k][j] -= rate * (g->l[i].weights->data[k][j]);
			}
			nn->l[i].biases->data[j] -= rate * (g->l[i].biases->data[j]);

		}
	}
}

void displayModel(model nn){//works
	for(u32 i = 0; i < nn->lc;i++){
		printf("layer %u:\n",i);
		displayMat(nn->l[i].weights);
		displayVec(nn->l[i].biases);
	}
}
