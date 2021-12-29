//순환적인 피보나치 수열 계산
int fib(int n){
    if(n==0) return 0;
    if(n==1) return 1;
    return (fib(n-1) + fib(n-2));
}

//반족적인 피보나치 수열 계산
int fib_iter(int n){
    if(n==0)return 0;
    if(n==1)return 1;

    int pp = 0, p = 1, result = 0;

    for(int i = 2; i <= n; i++){
        result = p + pp;
        pp = p;
        p = result;
    }
    return result;
}