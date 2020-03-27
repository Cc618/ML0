from random import random


def predict(x):
    '''
        Prediction by the network
    '''
    return a * x + b


def show():
    '''
        Displays the data in a graph
    '''
    import matplotlib.pyplot as plt

    # Blue = ground truth
    plt.plot(x_data, y_data, 'b.')

    # Red = prediction
    plt.plot(x_data, [predict(x) for x in x_data], 'r-')

    plt.axis([0, 1, 2, 3])
    plt.show()


target_a = .5
target_b = 2
print_freq = 25

# * Dataset
n = 20
# x data is within [0, 1)
x_data = [x / n for x in range(n)]
y_data = [target_a * x + target_b for x in x_data]

# With noise :
# noise_strength = 5e-2
# y_data = [target_a * x + target_b + random() * noise_strength for x in x_data]

# * Weigths
# Init weights with 'random values'
a = -.12
b = 0

# * Hyper parameters
learning_rate = 1e-1
epochs = 500

# * Training
avg_loss = 0
for e in range(epochs):
    loss = 0
    da = 0
    db = 0

    # For each tuple (x, y) in the dataset
    for x, y in zip(x_data, y_data):
        yi = predict(x)
        loss += (yi - y) ** 2

        # * Compute gradient
        da += x * (yi - y)
        db += yi - y

    da /= n
    db /= n
    loss /= n

    # * Back propagate
    a -= learning_rate * da
    b -= learning_rate * db

    avg_loss += loss
    if e != 0 and e % print_freq == 0:
        print(f'Epoch : {e:3d} Loss : {avg_loss / n:1.6f}')
        avg_loss = 0

print(f'a = {a:.2f} b = {b:2.2f}')

print(f'Target : {[f"{target_a * x + target_b:.2f}" for x in x_data]}')
print(f'Guess  : {[f"{predict(x):.2f}" for x in x_data]}')

show()
