#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "src/matrix.h"
#include "src/model.h"

void _resetcol(){
    printf("\033[0m");                      //ANSI colour code
}

void _setcol(uc r,uc g,uc b){
    printf("\033[38;2;%u;%u;%um",r,g,b);    //ANSI colour code
}


int main(){
	srand(time(NULL));
	descriptor arch = newDescriptor(3,2,4,1);//works with 3,2,2,1 but it fails more often
	model nn = newModel(arch);

	printf("setting up dataset\n");

	data_t data = newdataset(4, 2,1);
	data.inputs[0][0] = 0;
	data.inputs[0][1] = 0;
	data.inputs[1][0] = 1;
	data.inputs[1][1] = 0;
	data.inputs[2][0] = 0;
	data.inputs[2][1] = 1;
	data.inputs[3][0] = 1;
	data.inputs[3][1] = 1;

	data.outputs[0][0] = 0;
	data.outputs[1][0] = 1;
	data.outputs[2][0] = 1;
	data.outputs[3][0] = 0;

	randModel(nn);	//if we want to randomize the model
	//zeroModel(nn);	//if we want every value in the model to be 0
	float eps = 0.1;
	float rate = 0.5;

	f32 iniC = cost(nn,data);
	for(u32 i = 0; i <= 20000;i++){
		model grad = finite_diff(nn, data, eps);
		learn(nn, grad, rate);
		destroyModel(grad);
		if(i %1000 == 0)printf("%u\tcost : %f\n",i,cost(nn,data));
	}


	printf("[");_setcol(255,0,0);printf("performance review");_resetcol();printf("]\n");
	printf("initial cost is : %f finished cost is : %f\n",iniC,cost(nn,data));
	printf("Grade : ");
	if(cost(nn, data) <= 0.001){_setcol(0, 255, 0);printf("PASS\n");_resetcol();}
	if(cost(nn, data) >  0.001){_setcol(255, 0, 0);printf("FAIL\n");_resetcol();}

	HumanVerification(nn,data);

	destroydataset(data);
	destroyModel(nn);
	destroyDesc(arch);

  	return EXIT_SUCCESS;
}
