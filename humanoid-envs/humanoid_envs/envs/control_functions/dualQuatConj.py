# --------------------------------------
# Método para calcular o quatérnio dual conjugado
# Parâmetros:
# q: quatérnio dual
# Retorno:
# qr: quatérnio dual conjugado
# --------------------------------------
import numpy as np

from humanoid_envs.envs.control_functions.quatConj import quatConj


def dualQuatConj(q):
    a = np.array([q[0, 0], q[1, 0], q[2, 0], q[3, 0]]).reshape((4, 1))
    b = np.array([q[4, 0], q[5, 0], q[6, 0], q[7, 0]]).reshape((4, 1))
    qp = quatConj(a)
    qd = quatConj(b)
    qr = np.concatenate((qp, qd), axis=0)
    return qr
