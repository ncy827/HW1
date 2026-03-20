import os
import random
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

class RLEngine:
    def __init__(self, n, start, end, obstacles):
        self.n = n
        self.start, self.end = tuple(start), tuple(end)
        self.obstacles = [tuple(o) for o in obstacles]
        self.gamma = 0.9
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # 0:上, 1:下, 2:左, 3:右
        self.symbols = ['&uparrow;', '&downarrow;', '&leftarrow;', '&rightarrow;']

    def move(self, r, c, a):
        nr, nc = r + a[0], c + a[1]
        if 0 <= nr < self.n and 0 <= nc < self.n and (nr, nc) not in self.obstacles:
            return nr, nc
        return r, c

    # HW1-2: 隨機策略評估 (維持現狀)
    def solve_hw1_2(self):
        probs = np.random.dirichlet(np.ones(4), size=(self.n, self.n))
        V = np.zeros((self.n, self.n))
        for _ in range(100):
            new_V = np.copy(V)
            for r in range(self.n):
                for c in range(self.n):
                    if (r, c) == self.end or (r, c) in self.obstacles: continue
                    v_sum = 0
                    for i, a in enumerate(self.actions):
                        nr, nc = self.move(r, c, a)
                        v_sum += probs[r,c,i] * ((-1 if (nr, nc) != self.end else 0) + self.gamma * V[nr, nc])
                    new_V[r, c] = v_sum
            V = new_V
        P = [["" for _ in range(self.n)] for _ in range(self.n)]
        for r in range(self.n):
            for c in range(self.n):
                if (r, c) == self.end or (r, c) in self.obstacles: continue
                P[r][c] = self.symbols[np.argmax(probs[r,c])]
        return V.tolist(), P

    # HW1-3: 價值迭代 (修正箭頭與路徑一致性)
    def solve_hw1_3(self):
        V = np.zeros((self.n, self.n))
        for _ in range(100):
            new_V = np.copy(V)
            for r in range(self.n):
                for c in range(self.n):
                    if (r, c) == self.end or (r, c) in self.obstacles: continue
                    v_list = [(-1 if self.move(r,c,a)!=self.end else 0) + self.gamma*V[self.move(r,c,a)] for a in self.actions]
                    new_V[r, c] = max(v_list)
            V = new_V
        
        # 初始策略矩陣
        P = [["" for _ in range(self.n)] for _ in range(self.n)]
        for r in range(self.n):
            for c in range(self.n):
                if (r, c) == self.end or (r, c) in self.obstacles: continue
                q = [V[self.move(r, c, a)] for a in self.actions]
                P[r][c] = self.symbols[np.argmax(q)]

        # 追蹤路徑
        path = [self.start]
        curr = self.start
        visited = {self.start}
        for _ in range(self.n * self.n):
            if curr == self.end: break
            q = [V[self.move(curr[0], curr[1], a)] for a in self.actions]
            max_v = max(q)
            best_idxs = [i for i, v in enumerate(q) if v == max_v]
            
            # 隨機選一個最佳動作方向
            choice_idx = random.choice(best_idxs)
            best_a = self.actions[choice_idx]
            
            # 重要修正：強迫當前格子的箭頭指向路徑的下一個方向
            P[curr[0]][curr[1]] = self.symbols[choice_idx]
            
            curr = self.move(curr[0], curr[1], best_a)
            if curr in visited: break
            path.append(curr); visited.add(curr)
            
        return V.tolist(), P, path

@app.route('/')
def index(): return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    d = request.json
    s = RLEngine(d['n'], d['start'], d['end'], d['obstacles'])
    if d['mode'] == 'hw1-2':
        v, p = s.solve_hw1_2()
        return jsonify({'v': v, 'p': p})
    else:
        v, p, path = s.solve_hw1_3()
        return jsonify({'v': v, 'p': p, 'path': path})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
