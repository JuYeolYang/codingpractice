#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int main(void){
    int n, top = -1;
    char p[6];
    char command[5];
    scanf("%d", &n);
    getchar();
    int *list = (int*)malloc(sizeof(int) * n);

    for(int i = 0; i < n; i++){
        gets(p);
        if(strncmp("push", p, 4) == 0){
            top++;
            list[top] = atoi(&p[5]);
        }
        else if(strncmp("pop", p, 3) == 0){
            if(top == -1) printf("%d\n", top);
            else {
                printf("%d\n", list[top]);
                top--;
            }
        }
        else if(strncmp("size", p, 4) == 0){
            int temp = top + 1;
            printf("%d\n", temp);
        }
        else if(strncmp("empty", p, 5) == 0){
            if(top == -1) printf("%d\n", 1);
            else printf("%d\n", 0);
        }
        else if(strncmp("top", p, 3) == 0){
            if(top == -1) printf("%d\n", top);
            else printf("%d\n", list[top]);
        }
    }
    free(list);
}
