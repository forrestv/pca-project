// compile with: gcc -fopenmp rht.c -o rht -O3 -lm

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>

#define AC(m, i, j) *(m+((i)*w+(j)))

void sobel(int w, int h, unsigned char * img, unsigned char * result) {
    int i, j;
    double Gx, Gy, G;
#pragma omp parallel for default(shared) private(i, j, Gx, Gy, G)
    for(i = 1; i < h-1; i++) {
        for(j = 1; j < w-1; j++) {
            Gx = AC(img, i-1, j-1)*-1 + AC(img, i, j-1)*-2 + AC(img, i+1, j-1)*-1 +
                        AC(img, i-1, j+1)*1 + AC(img, i, j+1)*2 + AC(img, i+1, j+1)*1;
            Gy = AC(img, i-1, j-1)*-1 + AC(img, i-1, j)*-2 + AC(img, i-1, j+1)*-1 +
                        AC(img, i+1, j-1)*1 + AC(img, i+1, j)*2 + AC(img, i+1, j+1)*1;
            G = sqrt(Gx*Gx + Gy*Gy) * .2;
            AC(result, i, j) = G;
        }
    }
}

int main(int argc, char *argv[]) {
    // load image
    FILE * f = fopen(argv[1], "rb");
    char line[100]; assert(fgets(line, sizeof(line), f)); // skip first line
    int w, h, max; assert(fscanf(f, "%i %i %i", &w, &h, &max) == 3);
    unsigned char * img = malloc(h*w*sizeof(unsigned char));
    int i, j;
    for(i = 0; i < h; i++) {
        for(j = 0; j < w; j++) {
            assert(fscanf(f, "%hhi ", &img[i*w+j]) == 1);
        }
    }
    fclose(f);
    
    // do filtering
    unsigned char * result = malloc(h*w*sizeof(unsigned char));
    sobel(w, h, img, result);
    
    // save filtered image
    f = fopen(argv[2], "wb");
    fprintf(f, "P2\n%i %i\n%i\n", w, h, max);
    for(i = 0; i < h; i++) {
        for(j = 0; j < w; j++) {
            fprintf(f, "%i ", result[i*w+j]);
        }
        fprintf(f, "\n");
    }
}
