# Neural Network Framework

This project implements a neural network framework using C++ and Python. It is developed for educational purposes to explore the workings of neural network frameworks like PyTorch.

## Features

- **Backend**: C++ backend for efficient tensor operations.
- **Python Wrapper**: Utilizes ctypes for Python integration.

## Example Usage

```python
import src
import src.nn as nn
import src.optim as optim
import random
import math

random.seed(1)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        
        return out

epochs = 10

model = MyModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
loss_list = []

x_values = [0. ,  0.4,  0.8,  1.2,  1.6,  2. ,  2.4,  2.8,  3.2,  3.6,  4. ,
        4.4,  4.8,  5.2,  5.6,  6. ,  6.4,  6.8,  7.2,  7.6,  8. ,  8.4,
        8.8,  9.2,  9.6, 10. , 10.4, 10.8, 11.2, 11.6, 12. , 12.4, 12.8,
       13.2, 13.6, 14. , 14.4, 14.8, 15.2, 15.6, 16. , 16.4, 16.8, 17.2,
       17.6, 18. , 18.4, 18.8, 19.2, 19.6, 20.]

y_true = []
for x in x_values:
    y_true.append(math.pow(math.sin(x), 2))

for epoch in range(epochs):
    for x, target in zip(x_values, y_true):
        x = src.Tensor([[x]]).T
        target = src.Tensor([[target]]).T

        outputs = model(x)
        loss = criterion(outputs, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss[0]:.4f}')
    loss_list.append(loss[0])
```
# Contributing
Contributions are welcome! If you'd like to contribute to this project, feel free to fork it and submit a pull request with your changes.

**I have planned feature additions to extend functionality.**
