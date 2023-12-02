import numpy as np


# --------------------------------------
# Método para calcular o operador de Halmilton
# quatérnios
# Parâmetros:
# op: tipo do operador + ou -
# h - quatérnio
# Retorno:
# T: Matriz resultante
# --------------------------------------
def HamiltonOp(h, op):
    # quatérnio
    h1 = h[0, 0]
    h2 = h[1, 0]
    h3 = h[2, 0]
    h4 = h[3, 0]
    # Operador -
    H_ = np.array([[h1, -h2, -h3, -h4],
                   [h2, h1, h4, -h3],
                   [h3, -h4, h1, h2],
                   [h4, h3, -h2, h1]])
    # operador +
    H = np.array([[h1, -h2, -h3, -h4],
                  [h2, h1, -h4, h3],
                  [h3, h4, h1, -h2],
                  [h4, -h3, h2, h1]])
    # tipo do retorno
    if op == 1:
        T = H
    else:
        T = H_

    return T
