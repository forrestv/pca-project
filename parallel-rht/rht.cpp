// compile with: gcc -fopenmp rht.c -o rht -O3 -lm

#include <vector>
#include <random>
#include <cstdint>
#include <iostream>

#include <omp.h>

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>

template<typename T>
struct Array {
    int32_t w, h;
    std::vector<T> data;
    Array(int w, int h) : w(w), h(h), data(w*h) { }
    T const & operator()(int x, int y) const { return data.at(y*w + x); }
    T & operator()(int x, int y) { return data.at(y*w + x); }
    void fill(T value) {
        for(int y = 0; y < h; y++) {
            for(int x = 0; x < h; x++) {
                (*this)(x, y) = value;
            }
        }
    }
};

void sobel(Array<uint8_t> const & img, Array<uint8_t> & result) {
    int y, x;
    double Gx, Gy, G;
#pragma omp parallel for default(shared) private(y, x, Gx, Gy, G)
    for(y = 1; y < img.h-1; y++) {
        int k;
        for(x = 1; x < img.w-1; x++) {
            Gx = img(x-1, y-1)*-1 + img(x-1, y)*-2 + img(x-1, y+1)*-1 +
                        img(x+1, y-1)*1 + img(x+1, y)*2 + img(x+1, y+1)*1;
            Gy = img(x-1, y-1)*-1 + img(x, y-1)*-2 + img(x+1, y-1)*-1 +
                        img(x-1, y+1)*1 + img(x, y+1)*2 + img(x+1, y+1)*1;
            G = sqrt(Gx*Gx + Gy*Gy) * .2;
            result(x, y) = G > 30 ? 255 : 0;
        }
    }
}

struct Point {
    double x, y;
};

std::vector<Point> to_list(Array<uint8_t> const & edge) {
    std::vector<Point> res;
    for(int y = 1; y < edge.h-1; y++) {
        for(int x = 1; x < edge.w-1; x++) {
            if(edge(x, y)) {
                res.push_back(Point{static_cast<double>(x), static_cast<double>(y)});
            }
        }
    }
    return res;
}

struct Line {
    double angle;
    double offset;
    Point p1;
    Point p2;
    
    static Line from_two_points(Point a, Point b) {
        Line res = Line{
            atan2(b.y-a.y, b.x-a.x),
            (a.y*b.x - a.x*b.y)/sqrt(pow(b.y-a.y, 2) + pow(b.x-a.x, 2)),
            a, b,
        };
        if(res.angle < 0) {
            res.angle = M_PI + res.angle;
            res.offset = -res.offset;
        }
        return res;
    }
    
    void draw(Array<uint8_t> & img) const {
        if(1) {
            double cx = -sin(angle)*offset, cy = cos(angle)*offset;
            double dx = cos(angle), dy = sin(angle);
            for(int32_t i = -static_cast<int32_t>(img.w)-img.h; i < img.w+img.h; i++) {
                int32_t x = cx + dx * i;
                int32_t y = cy + dy * i;
                if(x >= 0 && y >= 0 && x < img.w && y < img.h) {
                    img(x, y) = 0;
                }
            }
        } else {
            for(double i = 0; i < 1; i += 1./img.w) {
                img(p1.x*i+p2.x*(1-i), p1.y*i+p2.y*(1-i)) = 0;
            }
        }
    }
};



std::vector<Line> rht(int w, int h, std::vector<Point> const & points) {
    double max_offset = sqrt(w*w+h*h);
    int acc_offset_size = max_offset/2;
    int acc_theta_size = acc_offset_size;
    
    std::vector<Line> res;
    
    Array<double> acc(acc_offset_size, acc_theta_size);
    acc.fill(0);
    #pragma omp parallel default(shared)
    {
        std::default_random_engine generator(omp_get_thread_num());
        while(true) {
            Point const & p1 = points[std::uniform_int_distribution<int>(0, points.size()-1)(generator)];
            Point const & p2 = points[std::uniform_int_distribution<int>(0, points.size()-1)(generator)];
            if(p1.x == p2.x && p1.y == p2.y) continue;
            
            Line l = Line::from_two_points(p1, p2);
            
            int theta_index = (acc_theta_size-1)*l.angle/M_PI;
            assert(theta_index >= 0 && theta_index < acc_theta_size);
            
            int offset_index = acc_offset_size*(l.offset/max_offset/2+.5);
            assert(offset_index >= 0 && offset_index < acc_offset_size);
            
            acc(offset_index, theta_index) += 1;
            double thres = 100;
            if(acc(offset_index, theta_index) == thres) {
                bool done = false;
                #pragma omp critical
                {
                    res.push_back(l);
                    //std::cout << res.size() << std::endl;
                    if(res.size() >= 400) {
                        done = true;
                    }
                }
                //std::cout << l.angle << " " << l.offset << std::endl;
                int area = 10;
                for(int32_t a = std::max(offset_index, area) - area; a < std::min(offset_index, acc_offset_size - area) + area; a++) {
                    for(int32_t b = std::max(theta_index, area) - area; b < std::min(theta_index, acc_theta_size - area) + area; b++) {
                        acc(a, b) = 2*thres;
                    }
                }
                if(done) break;
            }
        }
    }
    
    return res;
}

inline double get_time() {
    timespec x; assert(clock_gettime(CLOCK_MONOTONIC, &x) == 0);
    return x.tv_sec * 1. + x.tv_nsec * 1e-9;
}

int main(int argc, char *argv[]) {
    // load image
    FILE * f = fopen(argv[1], "rb");
    char line[100]; assert(fgets(line, sizeof(line), f)); // skip first line
    int w, h, max; assert(fscanf(f, "%i %i %i", &w, &h, &max) == 3);
    Array<uint8_t> img(w, h);
    int y, x;
    for(y = 0; y < h; y++) {
        for(x = 0; x < w; x++) {
            assert(fscanf(f, "%hhi ", &img(x, y)) == 1);
        }
    }
    fclose(f);
    
    double t1 = get_time();
    
    // do filtering
    Array<uint8_t> result(w, h);
    sobel(img, result);
    
    double t2 = get_time();
    
    auto l = to_list(result);
    
    double t3 = get_time();
    
    std::vector<Line> res = rht(w, h, l);
    
    double t4 = get_time();
    
    std::cout << t2-t1 << " " << t3-t2 << " " << t4-t3 << std::endl;
    
    for(auto x : res) {
        x.draw(img);
    }
    
    // save filtered image
    f = fopen(argv[2], "wb");
    fprintf(f, "P2\n%i %i\n%i\n", w, h, max);
    for(y = 0; y < h; y++) {
        for(x = 0; x < w; x++) {
            fprintf(f, "%i ", img(x, y));
        }
        fprintf(f, "\n");
    }
}
