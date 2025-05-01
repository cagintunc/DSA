
import numpy as np
import random

def random_matrix(n,m):
    result = np.ndarray((n, m))
    for row in range(n):
        result[row] = np.array([random.randint(-100, 100) for col in range(m)])
    return result


class RNN:
    def __init__(self):
        self.w1 = np.random.randn() * 0.1
        self.w2 = np.random.randn() * 0.1
        self.w3 = np.random.randn() * 0.1
        self.w4 = np.random.randn() * 0.1

        self.b1 = 0.0
        self.b2 = 0.0
        self.b3 = 0.0
        self.b4 = 0.0

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return 1.0 if x > 0 else 0.0

    def clip(self, val, min_val=-10, max_val=10):
        return max(min(val, max_val), min_val)

    def fit(self, X, y, batch_size=1, learning_rate=0.001, epochs=15):
        for epoch in range(epochs):
            loss = 0
            for i in range(0, len(y), batch_size):
                for j in range(batch_size):
                    idx = i + j
                    if idx >= len(y):
                        break
                    x1, x2 = X[0][idx], X[1][idx]
                    label = y[idx]

                    f1_input = self.w1 * x1 + self.b1
                    f1 = self.relu(f1_input)

                    f2_input = self.w3 * x2 + self.b3 + f1 * self.w2 + self.b2
                    f2 = self.relu(f2_input)

                    y_pred = f2 * self.w4 + self.b4
                    loss += (y_pred - label) ** 2

                    dy = 2 * (y_pred - label)

                    df2_dz = self.relu_derivative(f2_input)
                    df1_dz = self.relu_derivative(f1_input)

                    dydw4 = f2 * dy
                    dydb4 = 1 * dy

                    dydf2 = self.w4 * dy
                    dz2 = df2_dz

                    dydw3 = x2 * dydf2 * dz2
                    dydb3 = dydf2 * dz2
                    dydw2 = f1 * dydf2 * dz2

                    df1 = self.w2 * dydf2 * dz2
                    dydw1 = x1 * df1 * df1_dz
                    dydb1 = df1 * df1_dz

                    dydb2 = dydf2 * dz2

                    self.w1 = self.clip(self.w1 - learning_rate * dydw1)
                    self.w2 = self.clip(self.w2 - learning_rate * dydw2)
                    self.w3 = self.clip(self.w3 - learning_rate * dydw3)
                    self.w4 = self.clip(self.w4 - learning_rate * dydw4)

                    self.b1 = self.clip(self.b1 - learning_rate * dydb1)
                    self.b2 = self.clip(self.b2 - learning_rate * dydb2)
                    self.b3 = self.clip(self.b3 - learning_rate * dydb3)
                    self.b4 = self.clip(self.b4 - learning_rate * dydb4)

            print(f"Epoch {epoch + 1} - Loss {np.sqrt(loss)}")
        
    def predict(self, X):
        y = []
        x1_c, x2_c = X
        for i in range(len(x1_c)):
            x1, x2 = x1_c[i], x2_c[i]
            f1_input = self.w1 * x1 + self.b1
            f1 = self.relu(f1_input)
            f2_input = self.w3 * x2 + self.b3 + f1 * self.w2 + self.b2
            f2 = self.relu(f2_input)
            y_pred = f2 * self.w4 + self.b4
            if y_pred < 0.5:
                y_pred = 0
            else:
                y_pred = 1
            y.append(y_pred)
        return y
            

def get_sample(number_of_sample):
    X1, X2, y_c = [], [], []

    for i in range(number_of_sample):
        x_1 = random.randint(20, 50)
        x_2 = random.randint(-40, 50)
        y = (x_1*0.0438) + (x_2*1.2) + 5.43
        X1.append(x_1)
        X2.append(x_2)
        if y < 5:
            y_c.append(0)
        else:
            y_c.append(1)

    return [X1, X2, y_c]


x1, x2, y = get_sample(20000)
rnn = RNN()
rnn.fit(X=[x1,x2], y=y, batch_size=10, epochs=50)

test_x1, test_x2, test_y = get_sample(5000)
y_pred = rnn.predict(X=[test_x1, test_x2])

summ = 0
for i in range(len(y_pred)):
    if y_pred[i] == test_y[i]:
        summ += 1

print(f"\nAccuracy: {summ/len(y_pred)}") #0.96 accuracy





