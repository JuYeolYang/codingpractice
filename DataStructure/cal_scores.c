int scores[6] = {0,1,2,3,4,5};

int get_max_score(n){
    int i, largest;
    largest = scores[0];
    for (i = 1; i < 5; i++){
        if (scores[i] > largest){
            largest = scores[i];
        }
    }
    return largest;
}