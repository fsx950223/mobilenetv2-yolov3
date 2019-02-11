#include <iostream>
using namespace std;
struct A{
    int a=0;
};
class B{
    public:
        int b=0;
};

void change(A a,B b){
    a.a=123;
    b.b=123;
    cout<< a.a <<" "<<b.b<<endl;
}
int main(){
    A a;
    B b;
    change(a,b);
    cout<< a.a <<" "<<b.b<<endl;
    return 0;
}



