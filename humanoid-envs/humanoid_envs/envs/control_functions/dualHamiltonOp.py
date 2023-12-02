# --------------------------------------
# Método para calcular operador de Halmiton
# de um quatérnio dual
# Parâmetros:
# h: quatérnio dual
# op: tipo de operador + ou -
# Retorno:
# T: matriz resultante
# --------------------------------------
import numpy as np

from humanoid_envs.envs.control_functions.HamiltonOp import HamiltonOp


def dualHamiltonOp(h, op):
    # separa elementos do dual quatérnio
    h1 = h[0, 0]
    h2 = h[1, 0]
    h3 = h[2, 0]
    h4 = h[3, 0]
    h5 = h[4, 0]
    h6 = h[5, 0]
    h7 = h[6, 0]
    h8 = h[7, 0]

    # calculo de
    a = np.array([h1, h2, h3, h4]).reshape((4, 1))
    b = np.array([h5, h6, h7, h8]).reshape((4, 1))
    Hp = HamiltonOp(a, op)
    Hd = HamiltonOp(b, op)
    # matriz de zeros
    O = np.zeros((4, 4))
    # operador de Halmiton para quatérnios duais
    T = np.concatenate((np.concatenate((Hp, O), axis=1), np.concatenate((Hd, Hp), axis=1)), axis=0)
    return T
