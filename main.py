from Game import Game
import numpy as np

up = 0
left = 1
down = 2
right = 3


def proba(state, state0, action, game):
    x, y = game._id_to_position(state)
    x0, y0 = game._id_to_position(state0)
    if action == up:
        if x0 == x - 1 and y0 == y:
            return 1
        else:
            return 0
    if action == down:
        if x0 == x + 1 and y0 == y:
            return 1
        else:
            return 0
    if action == left:
        if x0 == x and y0 == y - 1:
            return 1
        else:
            return 0
    if action == right:
        if x0 == x and y0 == y + 1:
            return 1
        else:
            return 0

# Markov decision process:


def mdp(game):
    S = tuple(i for i in range(game.m * game.n))
    # up left down right
    A = (up, left, down, right)

    p = np.zeros((len(S), len(S), len(A)))
    for s in range(len(S)):
        for s0 in range(len(S)):
            for a in range(len(A)):
                p[s, s0, a] = proba(S[s], S[s0], A[a], game)

    r = np.zeros((len(S), len(S), len(A)))
    for s in range(len(S)):
        for s0 in range(len(S)):
            for a in range(len(A)):
                r[s, s0, a] = game.move(A[a])[1]
    return S, A, p, r


def policy_iteration(mdp, game, gamma, epsilon):
    S, A, p, r = mdp(game)
    # 1. Initialization
    v = np.zeros(len(S))
    pi = np.zeros(len(S))
    while (True):
        # 2. Policy Evaluation
        while (True):
            delta = 0
            for s in S:
                v_current = v[s]
                # Update v[s]:
                v[s] = sum([p[int(s), int(sp), int(
                    pi[int(s)])]*(r[int(s), int(sp), int(pi[int(s)])] + gamma * v[sp]) for sp in S])
                delta = max(delta, abs(v_current - v[s]))
            if delta < epsilon * (1-gamma)/(2*gamma):
                break
        # 3. Policy Improvement
        stable = True
        for s in S:
            q_best = v[s]
            for a in A:
                q = sum([p[int(s), int(sp), int(a)]
                         * (r[int(s), int(sp), int(a)] + gamma * v[sp]) for sp in S])
                if q > q_best:
                    q_best = q
                    pi[s] = a
                    stable = False
        if stable:
            return pi


def print_policy(pi, game):
    def action(i):
        return {
            0: 'up',
            1: 'left',
            2: 'down',
            3: 'right'
        }.get(i)
    for i in range(game.n):
        for j in range(game.m):
            print(action(pi[j*game.m+i]), end=' ')
        print()

if __name__ == '__main__':
    my_game = Game(10, 10)
    pi = policy_iteration(mdp, my_game, 0.9, 0.0001)
    print_policy(pi, my_game)
