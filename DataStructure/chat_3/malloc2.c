#include <stdio.h>
#include <stdlib.h>

typedef struct studentTag{
    char name[10];  // 문자배열로 된 이름
    int age;        // 나이를 나타내는 정수값
    double gpa;     // 평균평점을 나타내는 실수값
} student;

int main(void){
    student *s;

    s = (student *)malloc(sizeof(student));
    if(s == NULL){
        fprintf(stderr, "메모리가 부족해서 할당할 수 없습니다.\n");
        exit(1);
    }

    strcpy(s->name, "Park");
    s->age = 20;

    free(s);
    return 0;
}