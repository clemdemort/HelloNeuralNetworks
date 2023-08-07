#ifndef MATRIX_H
#define MATRIX_H
#include <math.h>


typedef unsigned int nlu;
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


nlf randnlf();

//allocates memory;
mat newMat(nlu width, nlu height);

void destroyMat(mat matrix);
void displayMat(mat matrix);
void displayMatCol(mat matrix);
void zeroMat(mat matrix);
void randMat(mat matrix);
nlf mat_at(mat m, nlu x,nlu y);
nlf vec_at(vec v, nlu x);
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




#endif