#ifndef NEURALLIB_H
#define NEURALLIB_H
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <sys/types.h>
#include <assert.h>

#define RANDCAP 1
#define NL_ASSERT assert


//neurallib unsigned integer
typedef unsigned int nlu;
//neurallib float
typedef float nlf;
typedef unsigned char uc;

//a matrix is always accessed like so : mat[x][y]
//x max is width and y max is height
typedef struct mat_s{
    nlu w,h;       //width and height
    nlf ** data;    //actual data
}mat_t;

typedef struct vec_s{
    nlu h;         //width 
    nlf * data;    //actual data
}vec_t;

typedef mat_t * mat;
typedef vec_t * vec;
#define mat_at(m, x, y) (m)->data[x][y]
#define vec_at(v, x) (v)->data[x]

nlf randnlf();

//allocates memory;
mat newMat(nlu width, nlu height);

void destroyMat(mat matrix);
void displayMat(mat matrix);
void displayMatCol(mat matrix);
void zeroMat(mat matrix);
void randMat(mat matrix);
//allocates memory
vec newVec(nlu height);
//allocates memory
vec vcpy(vec src);

void zeroVec(vec vector);
void randVec(vec vector);

void destroyVec(vec vector);
void displayVec(vec vector);
void displayVecCol(vec vector);
void forallVecElements(vec vector , nlf (*fun)(nlf));


//math

//allocates memory
vec MatrixVectorProduct(mat m, vec v);
//allocates memory
vec Vadd(vec v1,vec v2);

//neural network implementation

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

#define outputlayer(a) &(a)->layers[(a)->lc-1];
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

#ifndef NEURALLIB_IMPLEMENTATION
#define NEURALLIB_IMPLEMENTATION

nlf randnlf(){//works
	return (nlf)rand()/(nlf)RAND_MAX;
}
nlf randnlfcapped(nlu cap){//works
	return ((nlf)(rand() % (200000*cap)) - (nlf)(100000.0*cap))/100000.0f;
}

mat newMat(nlu width, nlu height){//works
    mat res = malloc(sizeof(mat_t));
    res->data = malloc(sizeof(nlf*)*width);
    for(nlu i = 0; i < width;i++)res->data[i] = calloc(height,sizeof(nlf));
    res->w = width;
    res->h = height;
    return res;
}

void zeroMat(mat matrix){//works
    for(nlu i = 0; i < matrix->w;i++){
        for(nlu j = 0; j < matrix->h;j++){
            mat_at(matrix, i, j) = 0;
        }
    }
}

void randMat(mat matrix){//works
    for(nlu i = 0; i < matrix->w;i++){
        for(nlu j = 0; j < matrix->h;j++){
            mat_at(matrix, i, j) = randnlfcapped(RANDCAP);
        }
    }
}

void destroyMat(mat matrix){//works
    for(nlu i = 0; i < matrix->w;i++)free(matrix->data[i]);
    free(matrix->data);
    free(matrix);
}


/*

    I want this kind of display:

            +-     -+
            | 1 0 0 |
            | 0 1 0 |
            | 0 0 1 |
            +-     -+

    we will assume all data is between 0 and 1 since thats what we will be dealing with when working with neural networks
*/
void displayMat(mat matrix){
    printf("+-");
    for(nlu x = 0; x < matrix->w;x++)printf("      ");
    printf("-+\n");
    for(nlu y = 0; y < matrix->h;y++){
        printf("| ");
        for(nlu x = 0; x < matrix->w;x++){
            printf("%f ",matrix->data[x][y]);
        }
        printf(" |\n");
    }
    printf("+-");
    for(nlu x = 0; x < matrix->w;x++)printf("      ");
    printf("-+\n");

}

void resetcol(){
    printf("\033[0m");                      //ANSI colour code
}
void setcol(uc r,uc g,uc b){
    printf("\033[48;2;%u;%u;%um",r,g,b);    //ANSI colour code
}

void setTXTcol(uc r,uc g,uc b){
    printf("\033[38;2;%u;%u;%um",r,g,b);    //ANSI colour code
}

nlf min(nlf i,nlf j){
	return ((i <= j)*i) + ((i > j)*j);
}
uc sign(nlf i){
	return (i >= 0);
} 
/*

    Will display a matrix with every element between 0 and 1 as shades of red and blue

*/
void displayMatCol(mat matrix){
    resetcol();
    printf("+-");
    for(nlu x = 0; x < matrix->w;x++)printf("  ");
    printf("-+\n");
    for(nlu y = 0; y < matrix->h;y++){
        printf("| ");
        for(nlu x = 0; x < matrix->w;x++){
            int pcol = 255*matrix->data[x][y];
			uc col = abs((int)min(pcol,255));
			setcol(sign(mat_at(matrix,x, y)) * col,0,(1-sign(mat_at(matrix,x, y))) *col);
            printf("  ");
        }
        resetcol();
        printf(" |\n");
    }
    printf("+-");
    for(nlu x = 0; x < matrix->w;x++)printf("  ");
    printf("-+\n");

}


vec newVec(nlu height){
    vec res = malloc(sizeof(vec_t));
    res->data = calloc(height,sizeof(nlf));
    res->h = height;
    return res;
}
vec vcpy(vec src){
    vec dest = newVec(src->h);
    for(nlu i = 0; i < src->h;i++){
        dest->data[i] = src->data[i];
    }
    return dest;
}

void zeroVec(vec vector){
    for(nlu i = 0; i < vector->h;i++){
        vector->data[i] = 0;
    }
}

void randVec(vec vector){
    for(nlu i = 0; i < vector->h;i++){
        vector->data[i] = randnlfcapped(RANDCAP);
    }
}

void destroyVec(vec vector){
    free(vector->data);
    free(vector);
}
void displayVec(vec vector){
    printf("+-");
    printf("     ");
    printf("-+\n");
    for(nlu y = 0; y < vector->h;y++){
        printf("| ");
        printf("%f",vector->data[y]);
        printf(" |\n");
    }
    printf("+-");
    printf("     ");
    printf("-+\n");
}


void displayVecCol(vec vector){
    printf("+-");
    printf("  ");
    printf("-+\n");
    for(nlu y = 0; y < vector->h;y++){
        printf("| ");
        int pcol = 255*vector->data[y];
		uc col = abs((int)min(pcol,255));
		setcol(sign(vec_at(vector, y)) * col,0,(1-sign(vec_at(vector, y))) *col);
        printf("  ");
        resetcol();
        printf(" |\n");
    }
    printf("+-");
    printf("  ");
    printf("-+\n");
}

void forallVecElements(vec vector , nlf (*fun)(nlf)){
    for(nlu i = 0; i < vector->h; i++){
        vector->data[i] = fun(vector->data[i]);
    }
}

vec MatrixVectorProduct(mat m, vec v){//correct
    NL_ASSERT(m->w == v->h);
    vec res = newVec(m->h);
    for(nlu x = 0;x < m->h;x++){
        for(nlu y = 0; y < v->h;y++){
            res->data[x] += mat_at(m, y, x) * vec_at(v, y);
        }
    }
    return res;
    return NULL;
}

//allocates memory
vec Vadd(vec v1,vec v2){
    NL_ASSERT(v1->h == v2->h);//the operation could technicaly be done but since it shouldn't happen it'll be an error here
    vec res = newVec(v1->h);
    for(nlu i = 0; i < v1->h;i++){
        res->data[i] = v1->data[i] + v2->data[i];
    }
    return res;
}



//sigmoid function
nlf sig(nlf x){
	return 1.f/(1.f + expf(-1.f*x));	
}

//derivative of sigmoid function
nlf Dsig(nlf x){
	return x*(1.f-x);//i want to die i was wrong about this for TWO WEEKS
}

nlf max(nlf a,nlf b){
	if(a > b)return a;
	return b;
}

nlf reLU(nlf x){
	return max(0,x);	
}

//creates a descriptor to create the neural network
//the first argument is the numbers of layers
//each argument you put next is the amount of neurons inside the layer
//make sure that if you specify 3 layers you add 3 arguments otherwise weird things might happen.
descriptor newDescriptor(nlu descSize, ...){//works
	
	descriptor arch;
	arch.descsize = descSize;
	arch.desc = malloc(sizeof(nlu)*arch.descsize);

	// Declaring pointer to the
    // argument list
    va_list ptr;
 
    // Initializing argument to the
    // list pointer
    va_start(ptr, descSize);
	
 
    for (nlu i = 0; i < descSize; i++) {
 
        nlu val = va_arg(ptr, nlu);
		if(val > 10000){printf("[newDescriptor WARNING] %u Layer is huge (%u neurons) are you sure you didn't missuse the function?\n ",i,val);}
		arch.desc[i] = val;
    }
 
    // End of argument list traversal
    va_end(ptr);
 
    return arch;
}

descriptor getDescriptor(model nn){//works
	descriptor arch;
	arch.descsize = nn->lc+1;
	arch.desc = malloc(arch.descsize*sizeof(nlu));
	arch.desc[0] = nn->l[0].weights->w;
	for(nlu i = 1; i < arch.descsize;i++){
		arch.desc[i] = nn->l[i-1].biases->h;
	}
	return arch;
}

void destroyDesc(descriptor arch){//works
	free(arch.desc);
}


//wc : weight count -> how many weights should each neuron have (e.g how many neurons/entries before)
//nc : neuron count -> how many neurons in the layer
layer newlayer(nlu wc,nlu nc){//works
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
	for(nlu i = 0; i < res->lc;i++){
		//i = 0 means the first hidden layer
		res->l[i] = newlayer(arch.desc[i], arch.desc[i+1]);
	}
	return res;
}

void zeroModel(model m){//works
	for(nlu i = 0; i < m->lc; i++){
		zeroVec(m->l[i].biases);
		zeroMat(m->l[i].weights);
	}
}

void randModel(model m){//works
	for(nlu i = 0; i < m->lc; i++){
		randVec(m->l[i].biases);
		randMat(m->l[i].weights);
	}
}

void destroyModel(model m){//works
	for(nlu i = 0; i < m->lc; i++){
		destroyLayer(m->l[i]);
	}
	free(m->l);
    free(m);
}

void forward(activations a, model m,vec vinput){//probably works (hard to test)
	//SHOULD PUT VERIFICATION TO SEE IF ACTIVATION IS CORRECTLY ALLOCATED
	NL_ASSERT(a->lc-1 == m->lc);
	vec vprev = vcpy(vinput);
	for(nlu i = 0;i < m->lc;i++){
		NL_ASSERT(vprev->h == m->l[i].weights->w);
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
	for(nlu i = 0; i < D.descsize; i++){
		a->layers[i].h = D.desc[i];
		a->layers[i].data = calloc(D.desc[i],sizeof(nlf));
	}
	return a;
}

void destroyActivations(activations a){//works
	for(nlu i = 0; i < a->lc; i++){
		free(a->layers[i].data);
	}

	free(a->layers);
	free(a);
}

void zeroActivations(activations a){//works
	for(nlu i = 0; i < a->lc; i++){
		for(nlu j = 0; j < a->layers[i].h;j++){
			a->layers[i].data[j] = 0;
		}
	}
}


void displayActivations(activations a){//works
	for(nlu i = 0; i < a->lc; i++){
		printf("layer %u :\n",i);
		for(nlu j = 0; j < a->layers[i].h; j++){
			printf(" %f,",a->layers[i].data[j]);
		}
		printf("\n");
	}
}

data_t newdataset(nlu entries,nlu inputs, nlu outputs){
	data_t res;
	res.entry_count = entries;
	res.input_length = inputs;
	res.output_length = outputs;
	res.inputs  = malloc(sizeof(nlf*)*entries);
	res.outputs = malloc(sizeof(nlf*)*entries);

	for(nlu i = 0; i < entries; i++){
		res.inputs[i]  = malloc(sizeof(nlf)*inputs);
		res.outputs[i] = malloc(sizeof(nlf)*outputs);
	}
	return res;
}

void destroydataset(data_t data){
	for(nlu i = 0; i < data.entry_count; i++){
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
		for(nlu i = 0; i < vinput->h;i++){
			printf(" %f",vinput->data[i]);
		}
		printf(" =");
		for(nlu k = 0; k < data.output_length;k++){
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
		for(nlu k = 0; k < data.output_length;k++){
			out->data[k][i] = resc->data[k];		
		}
		//cleaning
		free(vinput);

	}
	displayMatCol(out);
	destroyMat(out);
	destroyActivations(resA);
}

nlf cost(model m,data_t e){
	nlf res = 0.0f;
	descriptor D = getDescriptor(m);
	activations resA = newActivations(D);
	destroyDesc(D);
	for(size_t i = 0; i < e.entry_count; i++){		 	//for all entries
		vec vinput = malloc(sizeof(vec_t));	  			//creating the input vector
		vinput->h = e.input_length;						//
		vinput->data = e.inputs[i];		 				//putting the data inside

		forward(resA, m, vinput);				 	//forwarding the model with the input vector
		vec resc = outputlayer(resA);
		for(nlu k = 0; k < e.output_length;k++){
			//printf("resc->data[%u] = %f e.outputs[%u][%u] = %f\n",k,resc->data[k],i,k,e.outputs[i][k]);
			nlf d = resc->data[k] - e.outputs[i][k]; //calculating the difference			
			res += d*d;								 //squaring the total (IDK why but 3b1b said so) and adding it up
		}
		//cleaning
		free(vinput);

	}
	//getting the mean
	res /= (nlf)e.entry_count;
	destroyActivations(resA);
	return res;
}


//wip (testing)
void backpropagation(model g,model nn,data_t e){
	if(nn->l[0].weights->w != e.input_length || nn->l[nn->lc-1].biases->h != e.output_length){printf("input/output mismatch between model and dataset\n");}
	descriptor arch = getDescriptor(nn);
	activations GA = newActivations(arch);
	activations act = newActivations(arch);
	destroyDesc(arch);
	zeroModel(g);
	nlu n = e.entry_count;
	for(nlu i = 0; i < e.entry_count;i++){
		vec vinput = malloc(sizeof(vec_t));	  	//creating the input vector
		vinput->h = e.input_length;						//
		vinput->data = e.inputs[i];		 				//putting the data inside
		forward(act,nn, vinput);
		free(vinput);


		//create empty activations for G
		zeroActivations(GA);
		//calculate Dcost in output of G
		//this is the error
        vec outp = outputlayer(act);
        vec outpA = outputlayer(GA);
        for (nlu j = 0; j < outp->h; j++) {
			outpA->data[j] = 2*(outp->data[j] - e.outputs[i][j]);
			//printf("%f\n",outputlayer(GA)->data[j]);
		}

		for(nlu l = GA->lc-1; l > 0; l--){//for each layer starting by the end
			for(nlu j = 0; j < GA->layers[l].h;j++){//for each activation of that layer
				nlf a = act->layers[l].data[j];
                nlf da = GA->layers[l].data[j];
                nlf qa = Dsig(a);
				//printf("Layer = %u DA = %f\n",l,da);
				g->l[l-1].biases->data[j] += da*qa;
                for (size_t k = 0; k < GA->layers[l-1].h; k++) {
					//j = height
					//k = width
                    nlf pa = act->layers[l-1].data[k];
                    nlf w = nn->l[l-1].weights->data[k][j];
					g->l[l-1].weights->data[k][j] += da*qa*pa;
                    GA->layers[l-1].data[k] += da*qa*w;
                }
			}
		}
	}
	destroyActivations(act);

	//displayModel(g);
	for (size_t i = 0; i < g->lc; i++) {
    	for (size_t j = 0; j < g->l[i].weights->h; j++) {
	        for (size_t k = 0; k < g->l[i].weights->w; k++) {
				g->l[i].weights->data[k][j] /= (nlf)n;
            }
			g->l[i].biases->data[j] /= (nlf)n;
        }
    }
	destroyActivations(GA);
}

//inefficient but works
void finite_diff(model g, model nn, data_t t, nlf eps)
{
	//SHOULD PUT VERIFICATION TO SEE IF ACTIVATION IS CORRECTLY ALLOCATED
    nlf saved;
    nlf c = cost(nn, t);
	zeroModel(g);
	for(nlu i = 0; i < nn->lc;i++){//for every layer
		for(nlu j = 0; j < nn->l[i].weights->h;j++){//for every activation
			for(nlu k = 0; k < nn->l[i].weights->w;k++){//for every weight
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

void learn(model nn, model g, nlf rate){

	NL_ASSERT(nn->lc == g->lc);
	NL_ASSERT(nn->l[0].biases->h == g->l[0].biases->h);
	for(nlu i = 0; i < nn->lc;i++){//for every layer
		for(nlu j = 0; j < nn->l[i].weights->h;j++){//for every activation
			for(nlu k = 0; k < nn->l[i].weights->w;k++){//for every weight
				nn->l[i].weights->data[k][j] -= rate * (g->l[i].weights->data[k][j]);
			}
			nn->l[i].biases->data[j] -= rate * (g->l[i].biases->data[j]);

		}
	}
}

void displayModel(model nn){//works
	for(nlu i = 0; i < nn->lc;i++){
		printf("layer %u:\n",i);
		displayMatCol(nn->l[i].weights);
		displayVecCol(nn->l[i].biases);
	}
}


#endif
