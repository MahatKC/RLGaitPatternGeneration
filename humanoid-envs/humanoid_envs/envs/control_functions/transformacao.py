import numpy as np

from humanoid_envs.envs.control_functions.quatMult import quatMult


# -------------------------------------------
# Método para retornar a transformação
# dado a posição e orientação
# Parâmetros:
# p - vetor de translação do sistema base
# r - orientação
# Retorno:
# h0 - quatérnio dual de transformação
# -------------------------------------------
def transformacao(p, r):
    qp = r  # parte primaria
    qd = 0.5 * quatMult(p, r)  # parte dual
    # quatérnio dual de posição e orientação atual
    h0 = np.concatenate((qp, qd), axis=0)
    return h0
