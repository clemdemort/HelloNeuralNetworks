#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "src/matrix.h"

typedef float cpl[2];
cpl data[] = {
	{0,0},
	{1,2},
	{2,4},
	{3,6},
	{4,8},
};
#define data_count (sizeof(data)/sizeof(data[0]))

float randfloat(){
	return (float)rand()/(float)RAND_MAX;
}

float cost(float w,float b){
	float res = 0.0f;
	for(size_t i = 0; i < data_count; i++){
		float x = data[i][0];
		float y = (x*w) + b;
		float d = y - data[i][1];
		res += d*d;

	}
	res /= data_count;
	return res;
}

int main(){
	srand(time(0));
	float w = randfloat()*10.0f;
	float b = randfloat()*5.0f;
	printf("%f,%f\n",w,b);
	float eps = 1e-3;
	float rate = 1e-3;
	float c = cost(w,b);
	while(c > 0.000005){
		c = cost(w,b);
		float dcost = (cost(w + eps,b) - c)/eps;
		float dbias = (cost(w,b + eps) - c)/eps;
		w -= rate*dcost;
		b -= rate*dbias;
		//printf("cost : %f \tW : %f \tB: %f \n",cost(w,b),w,b);
	}
	printf("cost : %f \tW : %f \tB: %f \n",cost(w,b),w,b);

	//printing some result
	printf("W : %f\n",w);
	printf("B : %f\n",b);
	for(size_t i = 0; i < data_count; i++){
		float x = data[i][0];
		float y = x*w + b;
		printf("actual : %f \texpected : %f\n",y,data[i][1]);

	}
  	return EXIT_SUCCESS;
}
