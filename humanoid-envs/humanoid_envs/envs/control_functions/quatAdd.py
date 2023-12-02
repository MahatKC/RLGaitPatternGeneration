# --------------------------------------
# Método para calcular a soma de dois
# quatérnios
# Parâmetros:
# q1: primeiro quatérnio
# q2: segundo quatérnio
# Retorno:
# qr: quatérnio resultante
# --------------------------------------
import numpy as np


def quatAdd(q1, q2):
    # q = a + bi + cj + dk
    qr = np.zeros((4, 1))
    qr[0, 0] = q1[0, 0] + q2[0, 0]  # parte real
    qr[1, 0] = q1[1, 0] + q2[1, 0]  # imaginário i
    qr[2, 0] = q1[2, 0] + q2[2, 0]  # imaginário j
    qr[3, 0] = q1[3, 0] + q2[3, 0]  # imaginário k
    return qr
