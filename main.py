"""Author: Brandon Trabucco, Copyright 2019
Implements an iterative LQR algorithm for solving a random dynamical system.
"""


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def build_graph(state_size, control_size):
    graph = tf.Graph()
    with graph.as_default():
        F_t = tf.placeholder(name="F_t", dtype=tf.float32, shape=[state_size, state_size + control_size])
        f_t = tf.placeholder(name="f_t", dtype=tf.float32, shape=[state_size, 1])
        C_t = tf.placeholder(name="C_t", dtype=tf.float32, shape=[state_size + control_size, state_size + control_size])
        c_t = tf.placeholder(name="c_t", dtype=tf.float32, shape=[state_size + control_size, 1])
        V_t = tf.placeholder(name="V_t", dtype=tf.float32, shape=[state_size, state_size])
        v_t = tf.placeholder(name="v_t", dtype=tf.float32, shape=[state_size, 1])
        Q_t = C_t + tf.matmul(tf.matmul(tf.transpose(F_t), V_t), F_t)
        q_t = c_t + tf.matmul(tf.matmul(tf.transpose(F_t), V_t), f_t) + tf.matmul(tf.transpose(F_t), v_t)
        Q_xt_xt = Q_t[:state_size, :state_size]
        Q_xt_ut = Q_t[:state_size, state_size:]
        Q_ut_xt = Q_t[state_size:, :state_size]
        Q_ut_ut = Q_t[state_size:, state_size:]
        q_xt = q_t[:state_size, :]
        q_ut = q_t[state_size:, :]
        Q_ut_ut_inv = -tf.linalg.inv(Q_ut_ut)
        K_t = tf.matmul(Q_ut_ut_inv, Q_ut_xt)
        k_t = tf.matmul(Q_ut_ut_inv, q_ut)
        V_t_minus_one = Q_xt_xt + tf.matmul(Q_xt_ut, K_t) + tf.matmul(tf.transpose(K_t), Q_ut_xt) + tf.matmul(tf.matmul(tf.transpose(K_t), Q_ut_ut), K_t)
        v_t_minus_one = q_xt + tf.matmul(Q_xt_ut, k_t) + tf.matmul(tf.transpose(K_t), q_ut) + tf.matmul(tf.matmul(tf.transpose(K_t), Q_ut_ut), k_t)
    sess = tf.Session(graph=graph)
    def run_function(_F_t, _f_t, _C_t, _c_t, _V_t, _v_t):
        return sess.run([K_t, k_t, V_t_minus_one, v_t_minus_one], feed_dict={F_t: _F_t, C_t: _C_t, V_t: _V_t, 
            f_t: _f_t, c_t: _c_t, v_t: _v_t})
    return run_function


def main():

    state_size = 100
    control_size = 10
    trajectory_size = 100
    num_iterations = 100

    F_t = np.random.normal(0, .1, [state_size, state_size + control_size])
    f_t = np.random.normal(0, .1, [state_size, 1])

    C_t = np.eye(state_size + control_size)
    c_t = np.zeros([state_size + control_size, 1])

    linear_quadratic_regulator = build_graph(state_size, control_size)

    def forward_pass(initial_state, actions, F_t, f_t, C_t, c_t):
        states = [initial_state]
        cost = 0.0
        for action in actions[:-1]:
            state_control = np.concatenate([states[-1], action], 0)
            cost += 0.5 * state_control.T.dot(C_t).dot(state_control) + state_control.T.dot(c_t)
            states.append(F_t.dot(state_control) + f_t)
        return states, cost

    def backward_pass(states, state_size, F_t, f_t, C_t, c_t):
        better_actions = []
        V_t = np.zeros([state_size, state_size])
        v_t = np.zeros([state_size, 1])
        for state in reversed(states):
            K_t, k_t, V_t, v_t = linear_quadratic_regulator(F_t, f_t, C_t, c_t, V_t, v_t)
            better_actions = [K_t.dot(state) + k_t] + better_actions
        return better_actions

    initial_state = np.random.normal(0, 1, [state_size, 1])
    actions = [np.zeros([control_size, 1]) for i in range(trajectory_size)]

    for i in range(num_iterations):
        states, cumulative_cost = forward_pass(initial_state, actions, F_t, f_t, C_t, c_t)
        actions = backward_pass(states, state_size, F_t, f_t, C_t, c_t)
        print("Iteration {0} cost was {1}".format(i, cumulative_cost[0, 0]))


if __name__ == "__main__":

    main()