"""Online algorithms. All updates see only (context, chosen action, reward).

Numba compiles sequential loops without batching away online feedback.
See REPRODUCTION.md for decisions where the paper is ambiguous.
"""
import numpy as np
from numba import njit

@njit(cache=True)
def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-max(-700.0, min(700.0, v))))

@njit(cache=True)
def exp3_probs(logw, gamma):
    w = np.exp(logw - np.max(logw))
    return (1-gamma)*w / np.sum(w) + gamma / len(w)

@njit(cache=True)
def draw(p):
    u = np.random.random()
    s = 0.0
    for i in range(len(p)):
        s += p[i]
        if u < s:
            return i
    return len(p)-1

@njit(cache=True)
def action_probs(scores, gamma):
    p = np.full(len(scores), gamma/len(scores))
    p[np.argmax(scores)] += 1-gamma
    return p

@njit(cache=True)
def forward(x, w1, w2, sizes):
    m, k, hmax, d = w1.shape
    hidden = np.zeros((m, k, hmax))
    scores = np.zeros((m, k))
    active = np.flatnonzero(x)
    for j in range(m):
        for a in range(k):
            out = 0.0
            for h in range(sizes[j]):
                v = 0.0
                for f in active:
                    v += w1[j,a,h,f]*x[f]
                hidden[j,a,h] = sigmoid(v)
                out += hidden[j,a,h]*w2[j,a,h]
            scores[j,a] = sigmoid(out)
    return scores, hidden

@njit(cache=True)
def predict_one(x, active, w1, w2, sizes, j, a, scores, hidden):
    out = 0.0
    for h in range(sizes[j]):
        v = 0.0
        for f in active:
            v += w1[j,a,h,f]*x[f]
        z = sigmoid(v)
        hidden[j,a,h] = z
        out += z*w2[j,a,h]
    scores[j,a] = sigmoid(out)

@njit(cache=True)
def update_networks(x, action, reward, propensity, scores, hidden, w1, w2, sizes, rates):
    # Gradient descent on 1/2 (reward - prediction)^2; no biases (paper N=d*C+C).
    active = np.flatnonzero(x)
    for j in range(len(sizes)):
        q = scores[j, action]
        delta = (reward-q)*q*(1-q)
        step = rates[j]/propensity
        for h in range(sizes[j]):
            z = hidden[j,action,h]
            old_out = w2[j,action,h]
            for f in active:
                w1[j,action,h,f] += step*delta*old_out*z*(1-z)*x[f]
            w2[j,action,h] += step*delta*z

@njit(cache=True)
def neural_run(x, y, order, start, rounds, sizes, rates, variant, seed,
               gamma=.005, model_gamma=.1, drift_every=0,
               nb3_all=True, marginal=False):
    np.random.seed(seed)
    k = 7
    m = len(sizes)
    w1 = np.random.uniform(-.5, .5, (m,k,np.max(sizes),x.shape[1]))
    w2 = np.random.uniform(-.5, .5, (m,k,np.max(sizes)))
    logw = np.zeros((k if variant == 3 else 1, m))
    rewards = np.zeros(rounds, dtype=np.uint8)
    selections = np.zeros((k if variant == 3 else 1, m), dtype=np.int64)
    for t in range(rounds):
        idx = order[(start+t) % len(order)]
        row = x[idx]
        active = np.flatnonzero(row)
        scores = np.zeros((m,k))
        hidden = np.zeros((m,k,np.max(sizes)))
        selected = np.zeros(k, dtype=np.int64)
        model_p = np.zeros((len(logw),m))
        if variant == 1:
            for a in range(k):
                predict_one(row,active,w1,w2,sizes,0,a,scores,hidden)
            p = action_probs(scores[0], gamma)
        elif variant == 2:
            model_p[0] = exp3_probs(logw[0], model_gamma)
            j = draw(model_p[0]); selected[:] = j
            selections[0,j] += 1
            for a in range(k):
                predict_one(row,active,w1,w2,sizes,j,a,scores,hidden)
            p = action_probs(scores[j], gamma)
        else:
            combined = np.zeros(k)
            for a in range(k):
                model_p[a] = exp3_probs(logw[a], model_gamma)
                j = draw(model_p[a]); selected[a] = j
                selections[a,j] += 1
                predict_one(row,active,w1,w2,sizes,j,a,scores,hidden)
                combined[a] = scores[j,a]
            p = action_probs(combined, gamma)
        action = draw(p)
        label = (int(y[idx]) + (t//drift_every if drift_every else 0)) % k
        reward = 1.0 if action == label else 0.0
        rewards[t] = int(reward)
        propensity = p[action]
        if variant == 2 and marginal:
            propensity = 0.0
            for j in range(m):
                for a in range(k):
                    predict_one(row,active,w1,w2,sizes,j,a,scores,hidden)
                propensity += model_p[0,j]*action_probs(scores[j], gamma)[action]
        # All configurations learn from the played action; compute only the
        # additional predictions needed for that update, before any mutation.
        for j in range(m):
            predict_one(row,active,w1,w2,sizes,j,action,scores,hidden)
        update_networks(row, action, reward, propensity, scores, hidden, w1, w2, sizes, rates)
        if variant == 2:
            j = selected[0]
            logw[0,j] += model_gamma*reward/(m*model_p[0,j])
            logw[0] -= np.max(logw[0])
        elif variant == 3:
            for a in range(k):
                if nb3_all or a == action:
                    j = selected[a]
                    logw[a,j] += model_gamma*reward/(m*model_p[a,j])
                    logw[a] -= np.max(logw[a])
    return rewards, logw, selections

@njit(cache=True)
def baseline_run(x, y, order, start, rounds, algorithm, seed, gamma=.005,
                 drift_every=0, alpha=1.0, ts_scale=.1):
    """algorithm: 0 Banditron, 1 disjoint LinUCB, 2 Gaussian linear CTS.

    Baseline hyperparameters are reconstruction choices, absent in the paper.
    """
    np.random.seed(seed)
    k, d = 7, x.shape[1]
    weights = np.zeros((k,d))
    inv = np.zeros((k,d,d))
    b = np.zeros((k,d))
    mean = np.zeros((k,d))
    for a in range(k):
        inv[a] = np.eye(d)
    rewards = np.zeros(rounds, dtype=np.uint8)
    for t in range(rounds):
        idx = order[(start+t) % len(order)]
        row = x[idx].astype(np.float64)
        label = (int(y[idx]) + (t//drift_every if drift_every else 0)) % k
        if algorithm == 0:
            scores = weights @ row
            greedy = np.argmax(scores)
            p = action_probs(scores, gamma)
            action = draw(p)
        else:
            scores = np.zeros(k)
            for a in range(k):
                v = inv[a] @ row
                variance = max(0., np.dot(row,v))
                scores[a] = np.dot(mean[a],row)
                if algorithm == 1:
                    scores[a] += alpha*np.sqrt(variance)
                else:
                    scores[a] += ts_scale*np.sqrt(variance)*np.random.randn()
            action = np.argmax(scores)
        reward = 1.0 if action == label else 0.0
        rewards[t] = int(reward)
        if algorithm == 0:
            weights[greedy] -= row
            if reward:
                weights[action] += row / p[action]
        else:
            v = inv[action] @ row
            inv[action] -= np.outer(v,v)/(1+np.dot(row,v))
            b[action] += reward*row
            mean[action] = inv[action] @ b[action]
    return rewards
