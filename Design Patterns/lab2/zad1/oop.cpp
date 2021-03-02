#include <iostream>
#include <assert.h>
#include <stdlib.h>

struct Point{
    int x;
    int y;
};

class Shape{
    private:
        Point center_;
    public:
        virtual void draw()=0;
        void moveShape(int x, int y){
             printf("translating circle (%d,%d) -> ", center_.x, center_.y);
             center_.x += x;
             center_.y += y;
             printf("(%d,%d)\n", center_.x, center_.y);
        }
};

class Circle : public Shape{
    private:
        double radius_;
    public:
        virtual void draw(){
            std::cerr <<"in draw Circle\n";
        }
};

class Square : public Shape{
    private:
        double side_;
    public:
        virtual void draw(){
            std::cerr <<"in draw Rectangle\n";
        }
};



class Rhomb : public Shape{
    private:
        double side_;
    public:
        virtual void draw(){
            std::cerr <<"in draw Rectangle\n";
        }
};

/*
 * Mnogo jednostavniji klijentski kod.
 * Dinamički polimorfizam -> poziva se najspecifičnija implementacija 
 */
void drawShapes(Shape** shapes,int n){
    for(int i = 0; i < n; ++i){
       shapes[i] -> draw();
    }
}

void moveShapes(Shape** shapes,int n, Point* p){
    for(int i = 0; i < n; ++i){
       shapes[i] -> moveShape(p -> x, p -> y);
    }
}

int main(){
    Shape* shapes[5];
    shapes[0]=(Shape*)new Circle;
    shapes[1]=(Shape*)new Square;
    shapes[2]=(Shape*)new Square;
    shapes[3]=(Shape*)new Circle;
    shapes[4]=(Shape*)new Rhomb;

    drawShapes(shapes, 5);

    Point p;
    p.x = 3;
    p.y = 4;

    moveShapes(shapes, 5,  &p);

}