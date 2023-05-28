#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "src/matrix.h"





typedef float cpl[3];
cpl data[] = {
	{0,0,0},
	{0,1,1},
	{1,0,1},
	{1,1,1},
};
#define data_count (sizeof(data)/sizeof(data[0]))


float sig(float x){
	return 1.f/(1.f + expf(-x));	
}


float randfloat(){
	return (float)rand()/(float)RAND_MAX;
}

float cost(float w1,float w2,float b){
	float res = 0.0f;
	for(size_t i = 0; i < data_count; i++){
		float x1 = data[i][0];
		float x2 = data[i][1];
		float y = sig((x1*w1) + (x2*w2) + b);
		float d = y - data[i][2];
		res += d*d;

	}
	res /= data_count;
	return res;
}

int main(){
	srand(time(0));
	float w1 = randfloat()*10.0f;
	float w2 = randfloat()*10.0f;
	float b = randfloat()*5.0f;
	float c = cost(w1,w2,b);
	printf("w1 : %f, w2 : %f, B : %f , cost : %f\n",w1,w2,b,c);
	float eps = 1e-1;
	float rate = 1e-1;
	for(size_t i = 0; i < 1000000; i++){
		c = cost(w1,w2,b);
		float dw1 = (cost(w1 + eps,w2,b) - c)/eps;
		float dw2 = (cost(w1,w2+eps,b) - c)/eps;
		float db = (cost(w1,w2,b + eps) - c)/eps;
		w1 -= rate*dw1;
		w2 -= rate*dw2;
		b -= rate*db;
		//printf("cost : %f \tW1 : %f \tW2 : %f \tB: %f \n",cost(w1,w2,b),w1,w2,b);
	}
	printf("cost : %f \tW1 : %f \tW2 : %f \tB: %f \n",cost(w1,w2,b),w1,w2,b);

	//printing some result
	printf("W1 : %f\n",w1);
	printf("W2 : %f\n",w2);
	printf("B : %f\n",b);

	
	for(size_t i = 0; i < data_count; i++){
		float x1 = data[i][0];
		float x2 = data[i][1];
		float y = sig((x1*w1) + (x2*w2) + b);
		printf("actual : %f \texpected : %f\n",y,data[i][2]);

	}
	printf("cost : %f\n",cost(w1,w2,b));
  	return EXIT_SUCCESS;
}
