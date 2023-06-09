#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"

#define RANDCAP 5

f32 randf32(){
	return (f32)rand()/(f32)RAND_MAX;
}
f32 randf32capped(u32 cap){
	return ((f32)(rand() % (20000*cap)) - (f32)(10000*cap))/10000.0f;
}

mat newMat(u32 width, u32 height){
    mat res = malloc(sizeof(mat_t));
    res->data = malloc(sizeof(f32*)*width);
    for(u32 i = 0; i < width;i++)res->data[i] = calloc(height,sizeof(f32));
    res->w = width;
    res->h = height;
    return res;
}

void zeroMat(mat matrix){
    for(u32 i = 0; i < matrix->w;i++){
        for(u32 j = 0; j < matrix->h;j++){
            matrix->data[i][j] = 0;
        }
    }
}

void randMat(mat matrix){
    for(u32 i = 0; i < matrix->w;i++){
        for(u32 j = 0; j < matrix->h;j++){
            matrix->data[i][j] = randf32capped(RANDCAP);
        }
    }
}

void destroyMat(mat matrix){
    for(u32 i = 0; i < matrix->w;i++)free(matrix->data[i]);
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
    for(u32 x = 0; x < matrix->w;x++)printf("      ");
    printf("-+\n");
    for(u32 y = 0; y < matrix->h;y++){
        printf("| ");
        for(u32 x = 0; x < matrix->w;x++){
            printf("%f ",matrix->data[x][y]);
        }
        printf(" |\n");
    }
    printf("+-");
    for(u32 x = 0; x < matrix->w;x++)printf("      ");
    printf("-+\n");

}

void resetcol(){
    printf("\033[0m");                      //ANSI colour code
}
void setcol(uc r,uc g,uc b){
    printf("\033[48;2;%u;%u;%um",r,g,b);    //ANSI colour code
}

void setTXTcol(uc r,uc g,uc b){
    printf("\033[38;2;%u;%u;%um",r,g,b);    //ANSI colour code
}


/*

    Will display a matrix with every element between 0 and 1 as shades of grey

*/
void displayMatCol(mat matrix){
    resetcol();
    printf("+-");
    for(u32 x = 0; x < matrix->w;x++)printf("  ");
    printf("-+\n");
    for(u32 y = 0; y < matrix->h;y++){
        printf("| ");
        for(u32 x = 0; x < matrix->w;x++){
            uc col = 255*matrix->data[x][y];
            setcol(col,col,col);
            printf("  ");
        }
        resetcol();
        printf(" |\n");
    }
    printf("+-");
    for(u32 x = 0; x < matrix->w;x++)printf("  ");
    printf("-+\n");

}


vec newVec(u32 height){
    vec res = malloc(sizeof(vec_t));
    res->data = calloc(height,sizeof(f32));
    res->h = height;
    return res;
}
vec vcpy(vec src){
    vec dest = newVec(src->h);
    for(u32 i = 0; i < src->h;i++){
        dest->data[i] = src->data[i];
    }
    return dest;
}

void zeroVec(vec vector){
    for(u32 i = 0; i < vector->h;i++){
        vector->data[i] = 0;
    }
}

void randVec(vec vector){
    for(u32 i = 0; i < vector->h;i++){
        vector->data[i] = randf32capped(RANDCAP);
    }
}

void destroyVec(vec vector){
    free(vector->data);
    free(vector);
}
void displayVec(vec vector){
    printf("+-");
    printf("     ");
    printf("-+\n");
    for(u32 y = 0; y < vector->h;y++){
        printf("| ");
        printf("%f",vector->data[y]);
        printf(" |\n");
    }
    printf("+-");
    printf("     ");
    printf("-+\n");
}

void displayVecCol(vec vector){
    printf("+-");
    printf("  ");
    printf("-+\n");
    for(u32 y = 0; y < vector->h;y++){
        printf("| ");
        uc col = 255*vector->data[y];
        setcol(col,col,col);
        printf("  ");
        resetcol();
        printf(" |\n");
    }
    printf("+-");
    printf("  ");
    printf("-+\n");
}

void forallVecElements(vec vector , f32 (*fun)(f32)){
    for(u32 i = 0; i < vector->h; i++){
        vector->data[i] = fun(vector->data[i]);
    }
}

vec MatrixVectorProduct(mat m, vec v){
    if(m->w == v->h){
        vec res = newVec(m->h);
        for(u32 x = 0;x < m->h;x++){
            for(u32 y = 0; y < v->h;y++){
                res->data[x] += m->data[y][x] * v->data[y];
            }
        }
        return res;
    }else{
        fprintf(stderr,"[ERROR] matrix width is not equal to vector length, operation is impossible!\n");
    }
    return NULL;
}
//allocates memory
vec Vadd(vec v1,vec v2){
    if(v1->h == v2->h){//the operation could technicaly be done but since it shouldn't happen it'll be an error here
        vec res = newVec(v1->h);
        for(u32 i = 0; i < v1->h;i++){
            res->data[i] = v1->data[i] + v2->data[i];
        }
        return res;
    }else{
        fprintf(stderr,"[ERROR] vector 1 length is not equal to vector 2 length, operation is impossible!\n");
    }
    return NULL;
}
