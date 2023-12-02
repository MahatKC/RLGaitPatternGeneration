import numpy as np

from humanoid_envs.envs.control_functions.quatAdd import quatAdd
from humanoid_envs.envs.control_functions.quatMult import quatMult


def dualQuatMult(p, q):
    # separando p em parte primaria e dual
    pp = np.array([p[0, 0], p[1, 0], p[2, 0], p[3, 0]]).reshape((4, 1))
    pd = np.array([p[4, 0], p[5, 0], p[6, 0], p[7, 0]]).reshape((4, 1))
    # separando q em parte primaria e dual
    qp = np.array([q[0, 0], q[1, 0], q[2, 0], q[3, 0]]).reshape((4, 1))
    qd = np.array([q[4, 0], q[5, 0], q[6, 0], q[7, 0]]).reshape((4, 1))

    aux1 = quatMult(pp, qp)
    aux2 = quatAdd(quatMult(pp, qd), quatMult(pd, qp))
    # quatérnio dual resultante
    qr = np.concatenate((aux1, aux2), axis=0)
    return qr
