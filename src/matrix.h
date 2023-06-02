#ifndef MATRIX_H
#define MATRIX_H
#include <math.h>
#include "model.h"


typedef unsigned int u32;
typedef float f32;
typedef unsigned char uc;
typedef struct mat_s{
    u32 w,h;       //width and height
    f32 ** data;    //actual data
}mat_t;

typedef struct vec_s{
    u32 h;         //width 
    f32 * data;    //actual data
}vec_t;

typedef mat_t * mat;
typedef vec_t * vec;

//allocates memory;
mat newMat(u32 width, u32 height);
void destroyMat(mat matrix);
void displayMat(mat matrix);
void displayMatCol(mat matrix);
mat weightstomat(layer l);  //take the L

//allocates memory
vec newVec(u32 height);
void destroyVec(vec vector);
void displayVec(vec vector);
void displayVecCol(vec vector);
vec layertovec(layer l);
vec biastovec(layer l);
void forallVecElements(vec vector , f32 (*fun)(f32));

//math

//allocates memory
vec MatrixVectorProduct(mat m, vec v);
//allocates memory
vec Vadd(vec v1,vec v2);



#endif