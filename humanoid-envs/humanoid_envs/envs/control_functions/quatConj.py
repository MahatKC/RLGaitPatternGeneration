# --------------------------------------
# Método para calcular o conjugado de
# um quatérnio
# Parâmetros:
# q: quatérnio
# Retorno:
# qr: quatérnio resultante
# --------------------------------------
import numpy as np


def quatConj(q):
    # q = a + bi + cj + dk
    qr = np.zeros((4, 1))
    qr[0, 0] = q[0, 0]  # parte real
    qr[1, 0] = -q[1, 0]  # imaginário i
    qr[2, 0] = -q[2, 0]  # imaginário j
    qr[3, 0] = -q[3, 0]  # imaginário k
    return qr
