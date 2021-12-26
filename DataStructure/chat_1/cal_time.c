//프로그램 효율성 측정하는 방법
/*
clock()함수가 호출되 었을 때의 시스템 시각을 CLOCKS_PER_SEC 단위로 반환한다
따라서 시작할 때 clock()함수를 호출하여 start변수에 저장하고 
끝날 때 다시 clock()함수를 호출하여 stop변수에 기록한다음
(start - stop)에 CLOCKS_PER_SEC으로 나누어 준다
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void){
    clock_t start, stop;
    double duration;
    start = clock(); //측정 시작

    for(int i = 0; i < 1000000; i++); //의미 없는 반복 루프

    stop = clock(); //측정 종료
    duration = (double)(stop - start) / CLOCKS_PER_SEC;
    printf("수행시간은 %f초입니다. \n", duration);
    return 0;
}