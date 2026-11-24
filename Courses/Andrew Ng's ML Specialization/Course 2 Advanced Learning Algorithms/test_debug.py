def gradient_descent_step(w, grad, alpha):
    w = w + alpha * grad
    return w

w = 10
grad = 1
alpha = 0.1

for _ in range(5):
    w = gradient_descent_step(w, grad, alpha)

print(w)