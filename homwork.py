from math import e, log
import matplotlib.pyplot as plt

# Звдвем логистичесую функцию
def logistic_function(z):
    return 1 / (1 + e ** (-z))

# Прописываем функцию ошибки
def logistic_error(outputs, targets):
    outputs = [max(min(o, 1 - 1e-5), 1e-5) for o in outputs] 
    error = -sum(t * log(o) + (1 - t) * log(1 - o) for t, o in zip(targets, outputs)) / len(targets)
    return error

# Прописываем веса для определения принадлежности вводимых данных к классам 
weights = [0.1, -0.2, 0.3]

# Прописываем основную функцию, объединяющую наши признаки и веса в одно значение
def weighted_z(point):
    z = sum(w * x for w, x in zip(weights[:-1], point)) + weights[-1]
    return z

# Функция для вычисления прогноза модели для заданной точки
def forward(point):
    z = weighted_z(point)
    return logistic_function(z)

# Функция обучение модели: прогоняем заданные данные по эпохам через прямой проход(предыдущую функцию), обновляем весы.
def train(inputs, targets, lr, epochs):
    global weights
    for epoch in range(epochs):
        outputs = [forward(point) for point in inputs]
        errors = [o - t for o, t in zip(outputs, targets)]
        
        for j in range(len(weights) - 1):  # Обновление весов
            weights[j] -= lr * sum(e * inp[j] for e, inp in zip(errors, inputs)) / len(inputs)
        weights[-1] -= lr * sum(errors) / len(inputs)  # Обновление свободного члена

        if epoch % 10 == 0:  # Печать каждые 10 эпох
            print(f"Epoch {epoch}, Error: {logistic_error(outputs, targets)}")

# Далее проводим оценку точности, представляющую точность предсказаний модели
def accuracy(inputs, targets):
    predictions = [round(forward(point)) for point in inputs]
    correct = sum(1 for p, t in zip(predictions, targets) if p == t)
    return correct / len(targets)

# Прописываем данные в виде матриц
X1 = [
    [0.38, 1.42, 0.55, 1.34, 1.76, 1.62, 0.83, 0.84, 1.77, 1.06],  # x координаты
    [1.79, 0.54, 0.34, 0.678, 1.64, 0.92, 1.49, 0.3, 0.7, 0.99]   # y координаты
]

X2 = [
    [3.9, 6.14, 6.1, 2.11, 3.23, 1.62, 1.88],   # x координаты
    [4.93, 4.95, 0.97, 0.77, 0.43, 4.61, 0.25]  # y координаты
]

# Преобразуем в общий список точек
inputs = [(X1[0][i], X1[1][i]) for i in range(len(X1[0]))] + \
         [(X2[0][i], X2[1][i]) for i in range(len(X2[0]))]
targets = [0] * len(X1[0]) + [1] * len(X2[0])

# Задаем гиперпараметры 
lr = 0.16
num_epochs = 100

# Проводим обучение модели
train(inputs, targets, lr, num_epochs)

# Проводится оценка модели
model_accuracy = accuracy(inputs, targets)
print(f"Точность модели: {model_accuracy * 100:.2f}%")
print(f"Финальные веса: {weights}")

# Прописываем визуализацию
plt.figure(figsize=(8, 6))
plt.scatter(X1[0], X1[1], color='red', label='Class 0')
plt.scatter(X2[0], X2[1], color='green', label='Class 1')

x_vals = [min(X1[0] + X2[0]), max(X1[0] + X2[0])]
y_vals = [-(weights[0] * x + weights[-1]) / weights[1] for x in x_vals]
plt.plot(x_vals, y_vals, color='blue', label='Граница гиперплоскости')

plt.title('Классификация логической регрессии')
plt.xlabel('Признак 1')
plt.ylabel('Признак  2')
plt.legend()
plt.show()
