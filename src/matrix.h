#ifndef MATRIX_H
#define MATRIX_H

typedef unsigned int uint;
typedef float f32;
typedef unsigned char uc;
typedef struct mat_s{
    uint w,h;       //width and height
    f32 ** data;    //actual data
}mat_t;

typedef mat_t * mat;

mat zero(uint width, uint height);
void destroyMat(mat matrix);
void displayMat(mat matrix);
void displayMatCol(mat matrix);

#endif