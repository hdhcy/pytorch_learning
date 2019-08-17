'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/8/13 16:43
'''
import numpy as np

# y=wx+b


def compute_error_for_line_given_points(b, w, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (w * x + b))**2
    return totalError / float(len(points))


def step_gradient(b_current, w_current, points, learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2 / N) * (y - ((w_current * x) + b_current))
        w_gradient += -(2 / N) * x * (y - ((w_current * x) + b_current))

    new_b = b_current - (learningRate * b_gradient)
    new_w = w_current - (learningRate * w_gradient)
    return [new_b, new_w]


def gradient_descent_runner(
        points,
        starting_b,
        starting_w,
        learning_rate,
        num_iterations):
    b = starting_b
    w = starting_w
    for i in range(num_iterations):
        b, w = step_gradient(b, w, np.array(points), learning_rate)
    return [b, w]


def run():
    points = np.genfromtxt('data.csv', delimiter=',')
    learining_rate = 0.0001
    initial_b = 0  # initial y_intercept guess
    initial_w = 0  # initial slope guess
    num_interations = 1000
    print(
        'staring gradient descent at b={0},m={1},error={2}' .format(
            initial_b,
            initial_w,
            compute_error_for_line_given_points(
                initial_b,
                initial_w,
                points)))
    print('Running...')
    [b, w] = gradient_descent_runner(
        points, initial_b, initial_w, learining_rate, num_interations)

    print(
        'After {0} iterations b={1},m={2},error={3}' .format(
            num_interations,
            b,
            w,
            compute_error_for_line_given_points(
                b,
                w,
                points)))


if __name__ == '__main__':
    run()

'''
import numpy as np


# y = wx + b
def compute_error_for_line_given_points(b, w, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (w * x + b)) ** 2
    return totalError / float(len(points))


def step_gradient(b_current, w_current, points, learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2 / N) * (y - ((w_current * x) + b_current))
        w_gradient += -(2 / N) * x * (y - ((w_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = w_current - (learningRate * w_gradient)
    return [new_b, new_m]


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, np.array(points), learning_rate)
    return [b, m]


def run():
    points = np.genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0  # initial y-intercept guess
    initial_m = 0  # initial slope guess
    num_iterations = 1000
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}"
          .format(initial_b, initial_m,
                  compute_error_for_line_given_points(initial_b, initial_m, points))
          )
    print("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".
          format(num_iterations, b, m,
                 compute_error_for_line_given_points(b, m, points))
          )


if __name__ == '__main__':
    run()
'''
