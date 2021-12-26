//최고 성적을 구하는 프로그램
#define MAX_ELEMENTS 100
int scores[MAX_ELEMENTS]; //자료구조

int get_max_score(int n){
    int i, largest;
    largest = scores[0];
    for(i = 1; i<n; i++) largest = scores[i];

    return largest;
}