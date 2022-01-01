#14501-퇴사
'''
T : 상담 완료하는데 걸리는 일
P : 상담했을 때 받을 수 있는 금액
N : 상담 가능 일 수

반복 or 순환으로 풀어야 됨
'''
N = int(input())
T = [0]
P = [0]
max_pay = [0 for i in range(N+1)]
for i in range(N):
    t, p = map(int, input().split(' '))
    T.append(t)
    P.append(p)

for i in range(N, 0, -1):
    if((i + T[i] - 1) > N):
        max_pay[i] = 0
    elif((i + T[i] - 1) == N):
        max_pay[i] = P[i];        
    else:
        max_pay[i] = P[i] + max(max_pay[(i + T[i]):(N+1)])

print(max(max_pay))
