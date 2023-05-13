#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"


mat zero(uint width, uint height){
    mat res = malloc(sizeof(mat_t));
    res->data = malloc(sizeof(f32*)*width);
    for(uint i = 0; i < width;i++)res->data[i] = calloc(height,sizeof(f32));
    res->w = width;
    res->h = height;
    return res;
}

void destroyMat(mat matrix){
    for(uint i = 0; i < matrix->w;i++)free(matrix->data[i]);
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
    for(uint x = 0; x < matrix->w;x++)printf("      ");
    printf("-+\n");
    for(uint y = 0; y < matrix->h;y++){
        printf("| ");
        for(uint x = 0; x < matrix->w;x++){
            printf("%.3f ",matrix->data[x][y]);
        }
        printf(" |\n");
    }
    printf("+-");
    for(uint x = 0; x < matrix->w;x++)printf("      ");
    printf("-+\n");

}

void resetcol(){
    printf("\033[0m");                      //ANSI colour code
}
void setcol(uc r,uc g,uc b){
    printf("\033[48;2;%u;%u;%um",r,g,b);    //ANSI colour code
}

/*

    Will display a matrix with every element between 0 and 1 as shades of grey

*/
void displayMatCol(mat matrix){
    resetcol();
    printf("+-");
    for(uint x = 0; x < matrix->w;x++)printf("  ");
    printf("-+\n");
    for(uint y = 0; y < matrix->h;y++){
        printf("| ");
        for(uint x = 0; x < matrix->w;x++){
            uc col = 255*matrix->data[x][y];
            setcol(col,col,col);
            printf("  ");
        }
        resetcol();
        printf(" |\n");
    }
    printf("+-");
    for(uint x = 0; x < matrix->w;x++)printf("  ");
    printf("-+\n");

}
