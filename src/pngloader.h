#ifndef PNGLOAD_H
#define PNGLOAD_H

#include <png.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

typedef unsigned int uint;
//a color is loaded on 32 bits 8bits for RGBA 
//00000000 00000000 00000000 00000000 
//  red     green     blue    alpha
typedef unsigned int col; 

typedef struct img_s{
    //width in pixels
    uint w;
    //height in pixels
    uint h;
    //access like this I->data[x][y] (x max is w) (y max is h)
    col ** data;
}img_t;

typedef img_t * img;

img loadIMG(const char * file);
void destroyIMG(img I);


#endif 