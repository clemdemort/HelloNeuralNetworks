#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "src/matrix.h"
#include "src/model.h"

typedef f32 cpl[3];
cpl data[] = {
	{0,0,0},
	{0,1,1},
	{1,0,1},
	{1,1,0},
};
#define data_count (sizeof(data)/sizeof(data[0]))


f32 sig(f32 x){
	return 1.f/(1.f + expf(-x));	
}


f32 randf32(){
	return (f32)rand()/(f32)RAND_MAX;
}
/*
f32 cost(neuron n){
	f32 res = 0.0f;
	for(size_t i = 0; i < data_count; i++){
		f32 y = 0;
		for(u32 j = 0; j < n->wc;j++){
			f32 x = data[i][j];
			y += x * n->w[j];
		}
		y = sig(y + n->b);
		f32 d = y - data[i][2];
		res += d*d;

	}
	res /= data_count;
	return res;
}
void train(neuron n,f32 eps, f32 rate){
	f32 * dw = malloc(n->wc*sizeof(f32));
	f32 c = cost(n);
	for(u32 j = 0; j < n->wc;j++){
		f32 save = n->w[j];
		n->w[j] += eps;
		dw[j] = rate * ((cost(n)-c)/eps);
		n->w[j] = save;
	}
	f32 save  = n->b;
	n->b += eps;
	f32 db = rate * ((cost(n)-c)/eps);
	n->b = save;
	n->b -= db;
	for(u32 j = 0; j < n->wc;j++){
		n->w[j] -= dw[j];
	}
	free(dw);
	//printf("cost : %f \tW1 : %f \tW2 : %f \tB: %f \n",cost(n),n->w[0],n->w[1],n->b);
	//printf("%f\n",cost(w1,w2,b));
}*/
int main(){
	/*
	srand(time(0));
	neuron n = newNeuron(2);
	//printf("w1 : %f, w2 : %f, B : %f , cost : %f\n",w1,w2,b,c);
	f32 eps = 1e-1;
	f32 rate = 1e-1;

	for(size_t i = 0; i < data_count; i++){
		f32 x1 = data[i][0];
		f32 x2 = data[i][1];
		f32 y = sig((x1*n->w[0]) + (x2*n->w[1]) + n->b);
		printf("actual : %f \texpected : %f\n",y,data[i][2]);

	}
	printf("cost : %f\n",cost(n));

	u32 tc = 10000000;
	printf("training %u times\n",tc);
	for(u32 i = 0; i < tc; i++){
		train(n, eps, rate);
	}
	
	for(size_t i = 0; i < data_count; i++){
		f32 x1 = data[i][0];
		f32 x2 = data[i][1];
		f32 y = sig((x1*n->w[0]) + (x2*n->w[1]) + n->b);
		printf("actual : %f \texpected : %f\n",y,data[i][2]);

	}
	printf("cost : %f\n",cost(n));*/
	descriptor arch;
	arch.descsize = 3;
	arch.desc = malloc(sizeof(u32)*arch.descsize);
	arch.desc[0] = 2;
	arch.desc[1] = 2;
	arch.desc[2] = 1;
	model nn = newModel(arch);
	//manually training the model for fun (will do when compute function is done) 
	nn->l[0].n[0].a = 4.0f;



	vec test = layertovec(nn->l[0]);
	displayVec(test);
	forallVecElements(test,sig);
	displayVec(test);
	destroyVec(test);
	destroyModel(nn);
	free(arch.desc);
	
	printf("we do based math now: \n");
	vec v1 = newVec(3);
	mat m1 = newMat(3, 3);
	m1->data[0][0] = 2.0f;
	m1->data[1][1] = 2.0f;
	m1->data[2][2] = 2.0f;

	v1->data[0] = 1.0f;
	v1->data[1] = 2.0f;
	v1->data[2] = 3.0f;

	vec v2 = MatrixVectorProduct(m1, v1);

	displayVec(v1);
	displayMat(m1);
	if(v2 != NULL){
		displayVec(v2);
		vec v3 = Vadd(v1, v2);
		if(v3 != NULL)displayVec(v3);
		destroyVec(v3);
	}
	destroyVec(v1);
	destroyVec(v2);
	destroyMat(m1);
  	return EXIT_SUCCESS;
}
