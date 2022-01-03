#include <iostream>
#include <stdlib.h>
using namespace std;

class Simple{
    public:
        Simple(){
            cout<<"I'm simple constructor!"<<endl;
        }
};

int main(void){
    cout<<"case 1: ";
    Simple * sp1 = new Simple; // Simple메소드가 실행됨

    cout<<"case 2: ";
    Simple * sp2 = (Simple*)malloc(sizeof(Simple)*1); // 아무것도 실행되지 않음

    cout<<endl<<"end of main"<<endl;
    delete sp1;
    free(sp2);
    return 0;
}