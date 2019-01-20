"""Author: Brandon Trabucco, Copyright 2019
Implements an iterative LQR algorithm for solving a random dynamical system.
"""


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def build_graph(state_size, control_size):

    graph = tf.Graph()
    with graph.as_default():

        initial_state = tf.placeholder(name="initial_state", dtype=tf.float32, shape=[state_size, 1])
        actions = tf.placeholder(name="actions", dtype=tf.float32, shape=[None, control_size, 1])
        F = tf.placeholder(name="F", dtype=tf.float32, shape=[None, state_size, state_size + control_size])
        f = tf.placeholder(name="f", dtype=tf.float32, shape=[None, state_size, 1])
        C = tf.placeholder(name="C", dtype=tf.float32, shape=[None, state_size + control_size, state_size + control_size])
        c = tf.placeholder(name="c", dtype=tf.float32, shape=[None, state_size + control_size, 1])

        sequence_length = tf.shape(F)[0]

        x_array = tf.TensorArray(dtype=tf.float32, size=sequence_length, clear_after_read=False)
        u_array = tf.TensorArray(dtype=tf.float32, size=sequence_length).unstack(actions)
        F_array = tf.TensorArray(dtype=tf.float32, size=sequence_length, clear_after_read=False).unstack(F)
        f_array = tf.TensorArray(dtype=tf.float32, size=sequence_length, clear_after_read=False).unstack(f)
        C_array = tf.TensorArray(dtype=tf.float32, size=sequence_length, clear_after_read=False).unstack(C)
        c_array = tf.TensorArray(dtype=tf.float32, size=sequence_length, clear_after_read=False).unstack(c)

        def forward_condition(i, sequence_length, cost, x_array, u_array, F_array, f_array, C_array, 
                c_array, previous_state): 
            return i < sequence_length

        def forward_body(i, sequence_length, cost, x_array, u_array, F_array, f_array, C_array, 
                c_array, previous_state): 
            F_t, f_t, C_t, c_t = F_array.read(i), f_array.read(i), C_array.read(i), c_array.read(i)
            state_control = tf.concat([previous_state, u_array.read(i)], 0)
            cost = cost + (0.5 * tf.matmul(tf.matmul(tf.transpose(state_control), C_t), state_control) 
                + tf.matmul(tf.transpose(state_control), c_t))
            next_state = tf.matmul(F_t, state_control) + f_t
            x_array = x_array.write(i, previous_state)
            return (i + 1, sequence_length, cost, x_array, u_array, F_array, f_array, C_array, 
                c_array, next_state)

        forward_results = tf.while_loop(forward_condition, forward_body, 
            loop_vars=[tf.constant(0), sequence_length, tf.constant([[0.0]]), x_array, u_array, F_array, f_array, C_array, 
                c_array, initial_state])

        cost = forward_results[2]
        x_array = forward_results[3]
        u_array = tf.TensorArray(dtype=tf.float32, size=sequence_length)

        def backward_condition(i, state_size, control_size, x_array, u_array, 
                F_array, f_array, C_array, c_array, V_t, v_t): 
            return i >= 0

        def backward_body(i, state_size, control_size, x_array, u_array, 
                F_array, f_array, C_array, c_array, V_t, v_t): 
            F_t, f_t, C_t, c_t = F_array.read(i), f_array.read(i), C_array.read(i), c_array.read(i)
            Q_t = C_t + tf.matmul(tf.matmul(tf.transpose(F_t), V_t), F_t)
            q_t = c_t + tf.matmul(tf.matmul(tf.transpose(F_t), V_t), f_t) + tf.matmul(tf.transpose(F_t), v_t)
            Q_xt_xt, Q_xt_ut, Q_ut_xt, Q_ut_ut = (
                Q_t[:state_size, :state_size], Q_t[:state_size, state_size:], 
                Q_t[state_size:, :state_size], Q_t[state_size:, state_size:])
            q_xt, q_ut = q_t[:state_size, :], q_t[state_size:, :]
            Q_ut_ut_inv = -tf.linalg.inv(Q_ut_ut)
            K_t, k_t = tf.matmul(Q_ut_ut_inv, Q_ut_xt), tf.matmul(Q_ut_ut_inv, q_ut)
            V_t_minus_one = tf.reshape(Q_xt_xt + tf.matmul(Q_xt_ut, K_t) + tf.matmul(tf.transpose(K_t), Q_ut_xt) 
                + tf.matmul(tf.matmul(tf.transpose(K_t), Q_ut_ut), K_t), tf.shape(V_t))
            v_t_minus_one = tf.reshape(q_xt + tf.matmul(Q_xt_ut, k_t) + tf.matmul(tf.transpose(K_t), q_ut) 
                + tf.matmul(tf.matmul(tf.transpose(K_t), Q_ut_ut), k_t), tf.shape(v_t))
            u_array = u_array.write(i, tf.matmul(K_t, x_array.read(i)) + k_t)
            return (i - 1, state_size, control_size, x_array, u_array,
                F_array, f_array, C_array, c_array, V_t_minus_one, v_t_minus_one)

        backward_results = tf.while_loop(backward_condition, backward_body, 
            loop_vars=[sequence_length - 1, state_size, control_size, 
                x_array, u_array, F_array, f_array, C_array, c_array, 
                tf.zeros([state_size, state_size]), tf.zeros([state_size, 1])])

        states = x_array.stack()
        controls = backward_results[4].stack()

    sess = tf.Session(graph=graph)
    def linear_quadratic_regulator(_initial_state, _actions, _F, _f, _C, _c):
        return sess.run([cost, states, controls], feed_dict={
            initial_state: _initial_state, actions: _actions, F: _F, C: _C, f: _f, c: _c})

    return linear_quadratic_regulator


def main():

    state_size = 100
    control_size = 10
    trajectory_size = 100
    num_iterations = 100

    F_t = np.random.normal(0, .1, [state_size, state_size + control_size])
    f_t = np.random.normal(0, .1, [state_size, 1])
    F = np.tile(F_t[np.newaxis, ...], [trajectory_size, 1, 1])
    f = np.tile(f_t[np.newaxis, ...], [trajectory_size, 1, 1])

    C_t = np.eye(state_size + control_size)
    c_t = np.zeros([state_size + control_size, 1])
    C = np.tile(C_t[np.newaxis, ...], [trajectory_size, 1, 1])
    c = np.tile(c_t[np.newaxis, ...], [trajectory_size, 1, 1])

    linear_quadratic_regulator = build_graph(state_size, control_size)
    initial_state = np.random.normal(0, 1, [state_size, 1])
    actions = np.zeros([trajectory_size, control_size, 1])

    for i in range(num_iterations):
        cost, states, actions = linear_quadratic_regulator(initial_state, actions, F, f, C, c)
        print("Iteration {0} cost was {1}".format(i, cost[0, 0]))


if __name__ == "__main__":

    main()