# --------------------------------------
# Método para obter a rotação de um quatérnio dual
# Parâmetros:
# q: quatérnio dual
# Retorno:
# r: quatérnio de rotação (orientação)
# --------------------------------------
import numpy as np

from humanoid_envs.envs.control_functions.quatNorm import quatNorm


def getRotationDualQuat(q):
    normq = quatNorm(q)

    if normq != 0:
        q = q / (normq)

    r = np.array([q[0, 0], q[1, 0], q[2, 0], q[3, 0]]).reshape((4, 1))
    return r
