#include <stdio.h>
#include <stdlib.h>
#include <string.h>


int main(void){
    char p[101];
    gets(p);
    int word_count = 0;
    for (int i = 0; i < strlen(p); i++){
        if(p[i] == ' ') word_count += 1;
    }
    word_count += 1;
    //printf("word_cout : %d\n",word_count);
    char **lib = (char**)malloc(sizeof(char*) * (word_count));
    

    int top, bottom = -1, row = 0, column = 0;
    for(top = 0; top <= strlen(p); top++){
        if(p[top] == ' '){
            //printf("top = %d, bottom = %d ",top, bottom);
            *(lib + row) = (char*)malloc(sizeof(char) * (top - bottom));
            for(int j = bottom + 1; j < top; j++){
                *(*(lib + row) + column) = p[j];
                column++;
            }
            *(*(lib + row) + (top - bottom - 1)) = '\0';
            //printf("word : %s\n",*(lib + row));
            bottom = top;
            column = 0;
            row++;
        }
        else if(top == strlen(p)){
            //printf("top = %d, bottom = %d ",top, bottom);
            *(lib + row) = (char*)malloc(sizeof(char) * (top - bottom));
            for(int j = bottom + 1; j < top; j++){
                *(*(lib + row) + column) = p[j];
                column++;
            }
            *(*(lib + row) + (top - bottom - 1)) = '\0';
            //printf("word : %s\n",*(lib + row));
            break;
        }else continue;
    } 

    //for(int i = 0; i < word_count; i++) printf("%s\n",*(lib + i));

    //printf("\n");

    char *temp;
    for(int i = 0; i < word_count; i++){
        for(int j = 0; j < word_count - i - 1; j++){
            if( strlen(*(lib + j)) < strlen(*(lib + j + 1))){
            temp = *(lib + j);
            *(lib + j) = *(lib + j + 1);
            *(lib + j + 1) = temp;
            }
        }
    }

    //for(int i = 0; i < word_count; i++) printf("%s\n",*(lib + i));

    for(int i = 0; i < word_count; i++){
        for(int j = 0; j < word_count - i - 1; j++){
            if( strlen(*(lib + j)) == strlen(*(lib + j + 1))){
                if(strcmp(*(lib + j),*(lib + j + 1)) > 0 ){
                     temp = *(lib + j);
                    *(lib + j) = *(lib + j + 1);
                    *(lib + j + 1) = temp;
                }
            }
        }
    }
    

    for(int i = 0; i < word_count; i++) printf("%s\n",*(lib + i));
    
    free(temp);
    for(int i = 0; i < word_count; i++) free(lib[i]);
    free(lib);
    return 0;
}


