import numpy as np
import sys
sys.setrecursionlimit(100000)

def hungarian(matrixTF): # [966, 35]
    edges = np.argwhere(matrixTF)
    lnum, rnum = matrixTF.shape
    graph = [[] for _ in range(lnum)]
    for edge in edges:
        graph[edge[0]].append(edge[1])
    match = [-1 for _ in range(rnum)]
    vis = [-1 for _ in range(rnum)]

    def dfs(u):
        for v in graph[u]:
            if vis[v]: continue
            vis[v] = True
            if match[v] == -1 or dfs(match[v]):
                match[v] = u
                return True
        return False

    ans = 0
    for a in range(lnum):
        for i in range(rnum):
            vis[i] = False
        if dfs(a):
            ans += 1
    assign = np.zeros((lnum, rnum), dtype=bool)
    for i, m in enumerate(match):
        if m >= 0:
            assign[m, i] = True
    return ans, assign # 15, [966, 35]