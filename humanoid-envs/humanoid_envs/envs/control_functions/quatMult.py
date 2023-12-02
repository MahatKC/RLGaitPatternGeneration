import numpy as np


def quatMult(q0, q1):
    qr = np.zeros((4, 1))
    qr[0, 0] = q0[0, 0] * q1[0, 0] - q0[1, 0] * q1[1, 0] - q0[2, 0] * q1[2, 0] - q0[3, 0] * q1[3, 0]  # parte real
    qr[1, 0] = q0[0, 0] * q1[1, 0] + q0[1, 0] * q1[0, 0] + q0[2, 0] * q1[3, 0] - q0[3, 0] * q1[2, 0]  # imaginário i
    qr[2, 0] = q0[0, 0] * q1[2, 0] - q0[1, 0] * q1[3, 0] + q0[2, 0] * q1[0, 0] + q0[3, 0] * q1[1, 0]  # imaginário j
    qr[3, 0] = q0[0, 0] * q1[3, 0] + q0[1, 0] * q1[2, 0] - q0[2, 0] * q1[1, 0] + q0[3, 0] * q1[0, 0]  # imaginário k
    return qr
