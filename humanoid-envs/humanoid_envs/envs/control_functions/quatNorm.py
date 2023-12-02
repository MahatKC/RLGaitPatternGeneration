# --------------------------------------
# Método para calcular a norma de
# um quatérnio
# Parâmetros:
# q: quatérnio
# Retorno:
# qr: quatérnio resultante
# --------------------------------------
import math as mt


def quatNorm(q):
    normq = mt.sqrt(q[0, 0] * q[0, 0] + q[1, 0] * q[1, 0] + q[2, 0] * q[2, 0] + q[3, 0] * q[3, 0])
    return normq
