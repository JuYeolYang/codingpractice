#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void){
    char fp[100], sp[100];
    int fwc = 0, swc = 0;

    gets(fp);
    gets(sp);

    for (int i = 0; i < strlen(fp); i++){
        if(fp[i] == ' ') fwc++;
    }
    fwc++;

    for (int i = 0; i < strlen(sp); i++){
        if(sp[i] == ' ') swc++;
    }
    swc++;

    char **flib = (char**)malloc(sizeof(char*) * (fwc));
    char **slib = (char**)malloc(sizeof(char*) * (swc));

    int top, bottom = -1, row = 0, column = 0;
    for(top = 0; top <= strlen(fp); top++){
        if(fp[top] == ' '){
            //printf("top = %d, bottom = %d ",top, bottom);
            *(flib + row) = (char*)malloc(sizeof(char) * (top - bottom));
            for(int j = bottom + 1; j < top; j++){
                *(*(flib + row) + column) = fp[j];
                column++;
            }
            *(*(flib + row) + (top - bottom - 1)) = '\0';
            //printf("word : %s\n",*(flib + row));
            bottom = top;
            column = 0;
            row++;
        }
        else if(top == strlen(fp)){
            //printf("top = %d, bottom = %d ",top, bottom);
            *(flib + row) = (char*)malloc(sizeof(char) * (top - bottom));
            for(int j = bottom + 1; j < top; j++){
                *(*(flib + row) + column) = fp[j];
                column++;
            }
            *(*(flib + row) + (top - bottom - 1)) = '\0';
            //printf("word : %s\n",*(flib + row));
            break;
        }else continue;
    }
    //for(int i = 0; i < fwc; i++) printf("%s\n",*(flib + i));
    //printf("\n");



    top = 0, bottom = -1, row = 0, column = 0;
    for(top = 0; top <= strlen(sp); top++){
        if(sp[top] == ' '){
            //printf("top = %d, bottom = %d ",top, bottom);
            *(slib + row) = (char*)malloc(sizeof(char) * (top - bottom));
            for(int j = bottom + 1; j < top; j++){
                *(*(slib + row) + column) = sp[j];
                column++;
            }
            *(*(slib + row) + (top - bottom - 1)) = '\0';
            //printf("word : %s\n",*(slib + row));
            bottom = top;
            column = 0;
            row++;
        }
        else if(top == strlen(sp)){
            //printf("top = %d, bottom = %d ",top, bottom);
            *(slib + row) = (char*)malloc(sizeof(char) * (top - bottom));
            for(int j = bottom + 1; j < top; j++){
                *(*(slib + row) + column) = sp[j];
                column++;
            }
            *(*(slib + row) + (top - bottom - 1)) = '\0';
            //printf("word : %s\n",*(slib + row));
            break;
        }else continue;
    }
    //for(int i = 0; i < swc; i++) printf("%s\n",*(slib + i));
    //printf("\n");

    
}