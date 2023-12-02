import math as mt

import numpy as np

from humanoid_envs.envs.control_functions.dualQuatConj import dualQuatConj


def dualQuatDH(o, d, a, s, k):
    # --------------------------------------
    # Método para calcular o quatérnio dual que representa
    # o frame atual
    # Parâmetros:
    # o - rotação de theta em z
    # d - distancia d em z
    # a - deslocamento de a em x
    # s - rotação de alpha em x
    # Retorno:
    # xe: quatérnio de transformação
    # --------------------------------------
    # calculo dos parâmetros
    h1 = mt.cos(o / 2) * mt.cos(s / 2)
    h2 = mt.cos(o / 2) * mt.sin(s / 2)
    h3 = mt.sin(o / 2) * mt.sin(s / 2)
    h4 = mt.sin(o / 2) * mt.cos(s / 2)
    # criando o quaternio da parte primária

    # calculo do quatérnio dual
    h5 = -0.5 * (d * h4 + a * h2)
    h6 = 0.5 * (-d * h3 + a * h1)
    h7 = 0.5 * (d * h2 + a * h4)
    h8 = 0.5 * (d * h1 - a * h3)
    # criando o quaternio da parte dual
    xe = np.array([h1, h2, h3, h4, h5, h6, h7, h8]).reshape((8, 1))
    # retorna o conjugado do quatérnio dual caso k
    # seja igual a 1
    if k == 1:
        xe = dualQuatConj(xe)

    return xe
