//순환 - 팩토리얼 계산
int factorial(int n){
    if(n <= 1) return 1;
    else return (n * factorial(n - 1));
}

//반복-팩토리얼 계산
int factorial_iter(int n){
    int i, result = 1;
    for(i = 1; i <= n; i++) result = result * i;
    return result;
}