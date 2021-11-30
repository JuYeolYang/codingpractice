#include <stdio.h>
#include <stdlib.h>
#include <string.h>


int main(void){
	char **lib = (char**)malloc(sizeof(char*) * 3);
	int temp = 0;
	for(int i = 0; i < 3; i++){
		*(lib + i) = (char*)malloc(sizeof(char) * 3);
		strcpy(*(lib + i), "ab");
		*(*(lib + i) + 2) = '\0';
	}
	for(int i = 0; i < 3; i++) printf("%s",*(lib + i));
}