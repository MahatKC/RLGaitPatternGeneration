import numpy as np

from humanoid_envs.envs.control_functions.dualQuatConj import dualQuatConj
from humanoid_envs.envs.control_functions.dualQuatDH import dualQuatDH
from humanoid_envs.envs.control_functions.dualQuatMult import dualQuatMult
from humanoid_envs.envs.control_functions.globalVariables import GlobalVariables
from humanoid_envs.envs.control_functions.kinematicModel import KinematicModel
from humanoid_envs.envs.control_functions.transformacao import transformacao


def kinematicRobo(theta, hOrg, hP, tipo, CoM):
    # --------------------------------------
    # Método para calcular o a cinemática direta
    # do robô utilizando quatérnio dual
    # height é a altura do robô
    # L1 é o tamanho do link da bacia
    # L2 é a altura da bacia até o primeiro motor
    # --------------------------------------
    # variaveis globais-----------------------------------------------------------------------
    glob = GlobalVariables()
    hpi = glob.getHpi()
    L1 = glob.getL1()
    L2 = glob.getL2()
    MDH_right = glob.getMDH_right()
    MDH_left = glob.getMDH_left()
    # ------------------------------------------------------------------------------------------------
    # l = np.size(theta[:,0],0)
    # r = np.size(theta[:,1],0)

    # thetar = np.zeros((l,1))
    # thetal = np.zeros((r,1))
    thetar = theta[:, 0].reshape((6, 1))
    thetal = theta[:, 1].reshape((6, 1))

    # transformações da origem para a origem
    # da configuração inicial das 2 pernas
    hCoM_O0_rightLeg = dualQuatDH(hpi, -L2, -L1, 0.0, 0.0)
    # transformação do sist de coordenadas do centro de massa para a origem 0 da perna direita

    p = np.array([0.0, 0.0, 0.0, 0.0]).reshape((4, 1))
    n = np.array([0.0, 1.0, 0.0])
    realRb = np.array([np.cos(np.pi / 8.0)])
    rb = np.concatenate((realRb, np.sin(np.pi / 8.0) * n), axis=0).reshape((4, 1))
    hd = transformacao(p, rb)  # posição desejada
    hCoM_O0_rightLeg = dualQuatMult(hCoM_O0_rightLeg, hd)  # transformação da base até o pé

    hCoM_O0_leftLeg = dualQuatDH(hpi, -L2, L1, 0.0, 0.0)
    # transformação do sist de coordenadas do centro de massa para a origem 0 da perna esquerda

    p = np.array([0.0, 0.0, 0, 0]).reshape((4, 1))
    n = np.array([0.0, 1.0, 0.0])
    realRb = np.array([np.cos(-np.pi / 8.0)])
    rb = np.concatenate((realRb, np.sin(-np.pi / 8.0) * n), axis=0).reshape((4, 1))
    hd = transformacao(p, rb)  # posição desejada
    hCoM_O0_leftLeg = dualQuatMult(hCoM_O0_leftLeg, hd)  # transformação da base até o pé

    hB_O6a = dualQuatMult(hOrg, hP)  # transformação para auxiliar na localização do pé em contato com o chão

    if tipo == 1:  # tipo = 1 significa que a perna direita está apoiada
        hO6_O0 = KinematicModel(MDH_right, thetar, 1, 0)
        # transformação do sistema de coordenadas do link O6 par ao link O0 (do início da perna para o pé)
    else:
        hO6_O0 = KinematicModel(MDH_left, thetal, 1, 0)

    hB_O0 = dualQuatMult(hB_O6a, hO6_O0)
    # representa a base ou sistema global (hOrg + hp), ou seja, do sistema base para o sistema O0
    # hB_O0 = hO6_O0

    if tipo == 1:
        hB_CoM = dualQuatMult(hB_O0, dualQuatConj(hCoM_O0_rightLeg))
    else:
        hB_CoM = dualQuatMult(hB_O0, dualQuatConj(hCoM_O0_leftLeg))

    hr = hB_CoM  # a função retornará a orientação do CoM (em relação à base global)

    if CoM == 0:
        # transformação da base O0 para O6
        if tipo == 1:
            hB_O0 = dualQuatMult(hB_CoM, hCoM_O0_leftLeg)
            hO0_O6 = KinematicModel(MDH_left, thetal, 6, 1)  # transformação da base 0 até a base 6
            hr = dualQuatMult(hB_O0, hO0_O6)  # posição do pé suspenso (nesse caso, o esquerdo)
        else:
            hB_O0 = dualQuatMult(hB_CoM, hCoM_O0_rightLeg)
            hO0_O6 = KinematicModel(MDH_right, thetar, 6, 1)
            hr = dualQuatMult(hB_O0, hO0_O6)

    return hr
