#include "pngloader.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <png.h>
#include <string.h>
#include <assert.h>




//only loads png of type 6 or 2
img loadIMG(const char * file){
    png_structp	png_ptr;
    png_infop info_ptr;
    FILE * fp;
    png_uint_32 width;
    png_uint_32 height;
    int bit_depth;
    int color_type;
    int interlace_method;
    int compression_method;
    int filter_method;
    uint j;
    png_bytepp rows;
    fp = fopen (file, "rb");
    if (! fp) {
	fprintf (stderr,"Cannot open '%s': %s\n", file, strerror (errno));
    }
    png_ptr = png_create_read_struct (PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (! png_ptr) {
	fprintf (stderr,"Cannot create PNG read structure");
    }
    info_ptr = png_create_info_struct (png_ptr);
    if (! png_ptr) {
	fprintf (stderr,"Cannot create PNG info structure");
    }
    png_init_io (png_ptr, fp);
    png_read_png (png_ptr, info_ptr, 0, 0);
    png_get_IHDR (png_ptr, info_ptr, & width, & height, & bit_depth,
		  & color_type, & interlace_method, & compression_method,
		  & filter_method);
    rows = png_get_rows (png_ptr, info_ptr);
    printf ("Width is %d, height is %d color type is %d\n", width, height,color_type);
    int rowbytes;
    rowbytes = png_get_rowbytes (png_ptr, info_ptr);
    printf ("Row bytes = %d\n", rowbytes);

    //allocating the new image
    img out = malloc(sizeof(img_t));
    out->w = width;
    out->h = height;
    out->data = malloc(sizeof(col*) * out->w);
    for(int x = 0;x < out->w;x++){
        out->data[x] = malloc(sizeof(col)*out->h);
    }
    if(color_type == 2 || color_type == 6){

    for (j = 0; j < height; j++) {
	    uint i;
	    png_bytep row;
	    row = rows[j];

        int add;
        if(color_type == 2)add = 3;
        if(color_type == 6)add = 4;

	    for (i = 0; i < rowbytes; i+=add) {
	        col c = 0;
	        c += row[i+0] << 24;
	        c += row[i+1] << 16;
	        c += row[i+2] << 8;
            if(color_type == 6)c += row[i+3];
            out->data[i/add][j] = c;

	    }
    }
    }else{
        fprintf(stderr,"only RGB and RGBA are currently supported sorry!\n");
    }
    png_destroy_read_struct(&png_ptr,&info_ptr, NULL);
    fclose(fp);
    return out;
}
void destroyIMG(img I){
    for(uint i = 0; i < I->w; i++){
        free(I->data[i]);
    }
    free(I->data);
    free(I);
}