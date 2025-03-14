from micrograd.nn import MLP
from micrograd.engine import Value

x = [2.0, 3.0, 1.0]
mlp = MLP(3, [4, 4, 1])
mlp(x)

xs = [
    [2.0, 3.0, -1.0],
    [-3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [.0, 1.0, -1.0]
]

ys = [1.0, -1.0, -1.0, 1.0]

epochs = 10
lr = 0.01
for i in range(epochs):
  # forward pass
  y_preds = [mlp(xi) for xi in xs]
  loss = sum([(yout-ygt)**2 for yout, ygt in zip(y_preds, ys)], Value(0.0))
  print('Loss: ', loss.data)

  # baclward pass
  mlp.zero_grad()
  loss.backward()

  # update
  for p in mlp.parameters():
    p.data += -lr*p.grad


