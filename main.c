#include <stdio.h>
#include <stdlib.h>
#include "src/matrix.h"
int main(){
	uint size = 28;
  	mat test = zero(size,size);
  	printf("displaying matrix:\n");
  	for(uint i = 0; i < test->w; i++)test->data[i][i] = 1;
  	displayMatCol(test);
  	destroyMat(test);
  	return EXIT_SUCCESS;
}
