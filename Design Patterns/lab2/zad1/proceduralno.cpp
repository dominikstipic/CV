#include <iostream>
#include <assert.h>
#include <stdlib.h>

struct Point{
    int x; int y;
};

struct Shape{
    enum EType {circle, square, rhomb};
    EType type_;
};

struct Circle{
    Shape::EType type_;
    double radius_;
    Point center_;
};

struct Square{
    Shape::EType type_;
    double side_;
    Point center_;
};

struct Rhomb{
    Shape::EType type_;
    double a;
    double c; 
    double side1;
    double side2;
    Point center_;
};

void drawSquare(struct Square*){
std::cerr <<"in drawSquare\n";
}
void drawCircle(struct Circle*){
std::cerr <<"in drawCircle\n";
}
void drawRhomb(struct Rhomb*){
std::cerr <<"in drawRhomb\n";
}

void translateCircle(struct Circle* circle, int x, int y){
    Circle* c = (struct Circle*) circle;
    printf("translating circle (%d,%d) -> ", c -> center_.x, c -> center_.y);
    c -> center_.x += x;
    c -> center_.y += y;
    printf("(%d,%d)\n", c -> center_.x, c -> center_.y);
}

void translateSquare(struct Square* square, int x, int y){
    Square* s = (struct Square*) square;
    printf("translating square (%d,%d) -> ", s -> center_.x, s -> center_.y);
    s -> center_.x += x;
    s -> center_.y += y;
    printf("(%d,%d)\n", s -> center_.x, s -> center_.y);
}




void moveShapes(Shape** shapes, int n, int x, int y){
    for(int i = 0; i < n; ++i){
        struct Shape* s = shapes[i];
        switch(s -> type_){
        case Shape::square: 
            translateSquare((struct Square*)s, x, y);
            break;
        break;
        case Shape::circle : 
            translateCircle((struct Circle*)s, x,y);
            break;
        break;
        default:
            assert(0);
            exit(0);
    }
    }
}


//DrawShapes je nemobilna jer mora imati pristup kodu Shape -> zbog zastavica.
//Uvijek moraju biti skupa
void drawShapes(Shape** shapes, int n){
    for (int i=0; i<n; ++i){
        struct Shape* s = shapes[i];
        switch (s->type_){
        case Shape::square:
            drawSquare((struct Square*)s);
            break;
        case Shape::circle:
            drawCircle((struct Circle*)s);
            break;
        case Shape::rhomb:
            drawRhomb((struct Rhomb*)s);
            break;
        default:
            assert(0); 
            exit(0);
        }
}
}
int main(){
    Shape* shapes[5];
    shapes[0]=(Shape*)new Circle;
    shapes[0]->type_=Shape::circle;
    shapes[1]=(Shape*)new Square;
    shapes[1]->type_=Shape::square;
    shapes[2]=(Shape*)new Square;
    shapes[2]->type_=Shape::square;
    shapes[3]=(Shape*)new Circle;
    shapes[3]->type_=Shape::circle;
    shapes[4]=(Shape*)new Rhomb;
    shapes[4]->type_=Shape::rhomb;

    drawShapes(shapes, 5);
    int x = 5;
    int y = 3;
    moveShapes(shapes, 5,  x, y);

}