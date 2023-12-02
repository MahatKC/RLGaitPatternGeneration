import numpy as np

from humanoid_envs.envs.control_functions.dualQuatDH import dualQuatDH
from humanoid_envs.envs.control_functions.dualQuatMult import dualQuatMult


def KinematicModel(MDH, theta, N, foward):
    # --------------------------------------
    # Método para calcular o a cinemática direta
    # da cadeia cinemática
    # Parâmetros:
    # MDH: parâmetros de Denavit-Hartenberg [thetai di ai alphai]
    # N: numero de linhas da tabela de MDH
    # foward: indica se a cinemática vai ser foward ou backward
    # Retorno:
    # xe: cinemática direta
    # --------------------------------------
    # vetor de retorno

    # foward
    xe = np.array([1, 0, 0, 0, 0, 0, 0, 0]).reshape((8, 1))
    if foward == 1:
        for i in range(0, N):
            # separar parâmetros
            o = MDH[i, 0] + theta[i, 0]
            d = MDH[i, 1]
            a = MDH[i, 2]
            s = MDH[i, 3]
            # cálculo de xe atual
            x = dualQuatDH(o, d, a, s, 0)
            xe = dualQuatMult(xe, x)

    else:
        # backward
        for i in range(5, N - 2, -1):
            # separar parâmetros
            o = MDH[i, 0] + theta[i, 0]
            d = MDH[i, 1]
            a = MDH[i, 2]
            s = MDH[i, 3]
            # cálculo de xe atual
            x = dualQuatDH(o, d, a, s, 1)  # conjugado
            xe = dualQuatMult(xe, x)

    return xe
