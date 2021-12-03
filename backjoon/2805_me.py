N, M = map(int, input().split(' '))
tree_list = [int(n) for n in input().split(' ')]

bottom = min(tree_list) #가장 낮은 나무 길이

while True:
    length = sum(tree_list, -1*(bottom * N))#벌목한 길이
    if length < M:
        temp = M - length
        if temp % N != 0: # 벌목한 길이가 M보다 길 경우
            print(int(bottom - (temp / N + 1)))
            break
        else:#벌목한 길이와 M이 같을 경우
            print(int(bottom - temp / N))
            break
    else:
        tree_list.remove(bottom)
        bottom = min(tree_list)
        N -= 1
        
