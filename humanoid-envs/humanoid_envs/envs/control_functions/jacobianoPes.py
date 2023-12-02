import numpy as np

from humanoid_envs.envs.control_functions.dualQuatDH import dualQuatDH
from humanoid_envs.envs.control_functions.dualQuatMult import dualQuatMult
from humanoid_envs.envs.control_functions.globalVariables import GlobalVariables
from humanoid_envs.envs.control_functions.kinematicModel import KinematicModel


def jacobianoPes(theta, ha, xe, leg):
    # -----------------------------------------------------------
    # c�lculo das derivadas para cada vari�vel de controle
    # ----------------------------------------------------------
    glob = GlobalVariables()
    MDH_right = glob.getMDH_right()
    MDH_left = glob.getMDH_left()
    hpi = glob.getHpi()
    L1 = glob.getL1()
    L2 = glob.getL2()

    thetal = theta[:, 1].reshape((6, 1))
    thetar = theta[:, 0].reshape((6, 1))
    z = np.zeros((8, 1))

    # transforma��es da origem para a origem
    # da configura��o inicial das 2 pernas
    hCoM_O0_rightLeg = dualQuatDH(hpi, -L2, -L1, 0, 0)
    # transformação do sist de coordenadas do centro de massa para a origem 0 da perna direita
    hCoM_O0_leftLeg = dualQuatDH(hpi, -L2, L1, 0, 0)
    # transformação do sist de coord. do CoM para a origem 0 da perna esquerda
    if leg == 1:  # left leg
        hB_O6a = dualQuatMult(ha, hCoM_O0_leftLeg)
        h = hB_O6a
        z[0, 0] = 0
        z[1, 0] = h[1, 0] * h[3, 0] + h[0, 0] * h[2, 0]
        z[2, 0] = h[2, 0] * h[3, 0] - h[0, 0] * h[1, 0]
        z[3, 0] = (h[3, 0] ** 2 - h[2, 0] ** 2 - h[1, 0] ** 2 + h[0, 0] ** 2) / 2
        z[4, 0] = 0
        z[5, 0] = h[1, 0] * h[7, 0] + h[5, 0] * h[3, 0] + h[0, 0] * h[6, 0] + h[4, 0] * h[2, 0]
        z[6, 0] = h[2, 0] * h[7, 0] + h[6, 0] * h[3, 0] - h[0, 0] * h[5, 0] - h[4, 0] * h[1, 0]
        z[7, 0] = h[3, 0] * h[7, 0] - h[2, 0] * h[6, 0] - h[1, 0] * h[5, 0] + h[0, 0] * h[4, 0]

        j0 = dualQuatMult(z, xe)
        ##################################j1##################################
        h = dualQuatMult(hB_O6a, KinematicModel(MDH_left, thetal, 1, 1))
        z[0, 0] = 0
        z[1, 0] = h[1, 0] * h[3, 0] + h[0, 0] * h[2, 0]
        z[2, 0] = h[2, 0] * h[3, 0] - h[0, 0] * h[1, 0]
        z[3, 0] = (h[3, 0] ** 2 - h[2, 0] ** 2 - h[1, 0] ** 2 + h[0, 0] ** 2) / 2
        z[4, 0] = 0
        z[5, 0] = h[1, 0] * h[7, 0] + h[5, 0] * h[3, 0] + h[0, 0] * h[6, 0] + h[4, 0] * h[2, 0]
        z[6, 0] = h[2, 0] * h[7, 0] + h[6, 0] * h[3, 0] - h[0, 0] * h[5, 0] - h[4, 0] * h[1, 0]
        z[7, 0] = h[3, 0] * h[7, 0] - h[2, 0] * h[6, 0] - h[1, 0] * h[5, 0] + h[0, 0] * h[4, 0]

        j1 = dualQuatMult(z, xe)

        ##################################j2##################################
        h = dualQuatMult(hB_O6a, KinematicModel(MDH_left, thetal, 2, 1))

        z[0, 0] = 0
        z[1, 0] = h[1, 0] * h[3, 0] + h[0, 0] * h[2, 0]
        z[2, 0] = h[2, 0] * h[3, 0] - h[0, 0] * h[1, 0]
        z[3, 0] = (h[3, 0] ** 2 - h[2, 0] ** 2 - h[1, 0] ** 2 + h[0, 0] ** 2) / 2
        z[4, 0] = 0
        z[5, 0] = h[1, 0] * h[7, 0] + h[5, 0] * h[3, 0] + h[0, 0] * h[6, 0] + h[4, 0] * h[2, 0]
        z[6, 0] = h[2, 0] * h[7, 0] + h[6, 0] * h[3, 0] - h[0, 0] * h[5, 0] - h[4, 0] * h[1, 0]
        z[7, 0] = h[3, 0] * h[7, 0] - h[2, 0] * h[6, 0] - h[1, 0] * h[5, 0] + h[0, 0] * h[4, 0]

        j2 = dualQuatMult(z, xe)

        ##################################j3##################################
        h = dualQuatMult(hB_O6a, KinematicModel(MDH_left, thetal, 3, 1))

        z[0, 0] = 0
        z[1, 0] = h[1, 0] * h[3, 0] + h[0, 0] * h[2, 0]
        z[2, 0] = h[2, 0] * h[3, 0] - h[0, 0] * h[1, 0]
        z[3, 0] = (h[3, 0] ** 2 - h[2, 0] ** 2 - h[1, 0] ** 2 + h[0, 0] ** 2) / 2
        z[4, 0] = 0
        z[5, 0] = h[1, 0] * h[7, 0] + h[5, 0] * h[3, 0] + h[0, 0] * h[6, 0] + h[4, 0] * h[2, 0]
        z[6, 0] = h[2, 0] * h[7, 0] + h[6, 0] * h[3, 0] - h[0, 0] * h[5, 0] - h[4, 0] * h[1, 0]
        z[7, 0] = h[3, 0] * h[7, 0] - h[2, 0] * h[6, 0] - h[1, 0] * h[5, 0] + h[0, 0] * h[4, 0]

        j3 = dualQuatMult(z, xe)

        ##################################j4##################################
        h = dualQuatMult(hB_O6a, KinematicModel(MDH_left, thetal, 4, 1))

        z[0, 0] = 0
        z[1, 0] = h[1, 0] * h[3, 0] + h[0, 0] * h[2, 0]
        z[2, 0] = h[2, 0] * h[3, 0] - h[0, 0] * h[1, 0]
        z[3, 0] = (h[3, 0] ** 2 - h[2, 0] ** 2 - h[1, 0] ** 2 + h[0, 0] ** 2) / 2
        z[4, 0] = 0
        z[5, 0] = h[1, 0] * h[7, 0] + h[5, 0] * h[3, 0] + h[0, 0] * h[6, 0] + h[4, 0] * h[2, 0]
        z[6, 0] = h[2, 0] * h[7, 0] + h[6, 0] * h[3, 0] - h[0, 0] * h[5, 0] - h[4, 0] * h[1, 0]
        z[7, 0] = h[3, 0] * h[7, 0] - h[2, 0] * h[6, 0] - h[1, 0] * h[5, 0] + h[0, 0] * h[4, 0]

        j4 = dualQuatMult(z, xe)

        ##################################j5##################################
        h = dualQuatMult(hB_O6a, KinematicModel(MDH_left, thetal, 5, 1))

        z[0, 0] = 0
        z[1, 0] = h[1, 0] * h[3, 0] + h[0, 0] * h[2, 0]
        z[2, 0] = h[2, 0] * h[3, 0] - h[0, 0] * h[1, 0]
        z[3, 0] = (h[3, 0] ** 2 - h[2, 0] ** 2 - h[1, 0] ** 2 + h[0, 0] ** 2) / 2
        z[4, 0] = 0
        z[5, 0] = h[1, 0] * h[7, 0] + h[5, 0] * h[3, 0] + h[0, 0] * h[6, 0] + h[4, 0] * h[2, 0]
        z[6, 0] = h[2, 0] * h[7, 0] + h[6, 0] * h[3, 0] - h[0, 0] * h[5, 0] - h[4, 0] * h[1, 0]
        z[7, 0] = h[3, 0] * h[7, 0] - h[2, 0] * h[6, 0] - h[1, 0] * h[5, 0] + h[0, 0] * h[4, 0]

        j5 = dualQuatMult(z, xe)
    else:
        hB_O6a = dualQuatMult(ha, hCoM_O0_rightLeg)
        h = hB_O6a
        # h = [1 0 0 0 0 0 0 0]'
        z[0, 0] = 0
        z[1, 0] = h[1, 0] * h[3, 0] + h[0, 0] * h[2, 0]
        z[2, 0] = h[2, 0] * h[3, 0] - h[0, 0] * h[1, 0]
        z[3, 0] = (h[3, 0] ** 2 - h[2, 0] ** 2 - h[1, 0] ** 2 + h[0, 0] ** 2) / 2
        z[4, 0] = 0
        z[5, 0] = h[1, 0] * h[7, 0] + h[5, 0] * h[3, 0] + h[0, 0] * h[6, 0] + h[4, 0] * h[2, 0]
        z[6, 0] = h[2, 0] * h[7, 0] + h[6, 0] * h[3, 0] - h[0, 0] * h[5, 0] - h[4, 0] * h[1, 0]
        z[7, 0] = h[3, 0] * h[7, 0] - h[2, 0] * h[6, 0] - h[1, 0] * h[5, 0] + h[0, 0] * h[4, 0]

        j0 = dualQuatMult(z, xe)
        ##################################j1##################################
        h = dualQuatMult(hB_O6a, KinematicModel(MDH_right, thetar, 1, 1))
        # h = [1 0 0 0 0 0 0 0]'
        z[0, 0] = 0
        z[1, 0] = h[1, 0] * h[3, 0] + h[0, 0] * h[2, 0]
        z[2, 0] = h[2, 0] * h[3, 0] - h[0, 0] * h[1, 0]
        z[3, 0] = (h[3, 0] ** 2 - h[2, 0] ** 2 - h[1, 0] ** 2 + h[0, 0] ** 2) / 2
        z[4, 0] = 0
        z[5, 0] = h[1, 0] * h[7, 0] + h[5, 0] * h[3, 0] + h[0, 0] * h[6, 0] + h[4, 0] * h[2, 0]
        z[6, 0] = h[2, 0] * h[7, 0] + h[6, 0] * h[3, 0] - h[0, 0] * h[5, 0] - h[4, 0] * h[1, 0]
        z[7, 0] = h[3, 0] * h[7, 0] - h[2, 0] * h[6, 0] - h[1, 0] * h[5, 0] + h[0, 0] * h[4, 0]

        j1 = dualQuatMult(z, xe)

        ##################################j2##################################
        h = dualQuatMult(hB_O6a, KinematicModel(MDH_right, thetar, 2, 1))

        z[0, 0] = 0
        z[1, 0] = h[1, 0] * h[3, 0] + h[0, 0] * h[2, 0]
        z[2, 0] = h[2, 0] * h[3, 0] - h[0, 0] * h[1, 0]
        z[3, 0] = (h[3, 0] ** 2 - h[2, 0] ** 2 - h[1, 0] ** 2 + h[0, 0] ** 2) / 2
        z[4, 0] = 0
        z[5, 0] = h[1, 0] * h[7, 0] + h[5, 0] * h[3, 0] + h[0, 0] * h[6, 0] + h[4, 0] * h[2, 0]
        z[6, 0] = h[2, 0] * h[7, 0] + h[6, 0] * h[3, 0] - h[0, 0] * h[5, 0] - h[4, 0] * h[1, 0]
        z[7, 0] = h[3, 0] * h[7, 0] - h[2, 0] * h[6, 0] - h[1, 0] * h[5, 0] + h[0, 0] * h[4, 0]

        j2 = dualQuatMult(z, xe)

        ##################################j3##################################
        h = dualQuatMult(hB_O6a, KinematicModel(MDH_right, thetar, 3, 1))

        z[0, 0] = 0
        z[1, 0] = h[1, 0] * h[3, 0] + h[0, 0] * h[2, 0]
        z[2, 0] = h[2, 0] * h[3, 0] - h[0, 0] * h[1, 0]
        z[3, 0] = (h[3, 0] ** 2 - h[2, 0] ** 2 - h[1, 0] ** 2 + h[0, 0] ** 2) / 2
        z[4, 0] = 0
        z[5, 0] = h[1, 0] * h[7, 0] + h[5, 0] * h[3, 0] + h[0, 0] * h[6, 0] + h[4, 0] * h[2, 0]
        z[6, 0] = h[2, 0] * h[7, 0] + h[6, 0] * h[3, 0] - h[0, 0] * h[5, 0] - h[4, 0] * h[1, 0]
        z[7, 0] = h[3, 0] * h[7, 0] - h[2, 0] * h[6, 0] - h[1, 0] * h[5, 0] + h[0, 0] * h[4, 0]

        j3 = dualQuatMult(z, xe)

        ##################################j4##################################
        h = dualQuatMult(hB_O6a, KinematicModel(MDH_right, thetar, 4, 1))

        z[0, 0] = 0
        z[1, 0] = h[1, 0] * h[3, 0] + h[0, 0] * h[2, 0]
        z[2, 0] = h[2, 0] * h[3, 0] - h[0, 0] * h[1, 0]
        z[3, 0] = (h[3, 0] ** 2 - h[2, 0] ** 2 - h[1, 0] ** 2 + h[0, 0] ** 2) / 2
        z[4, 0] = 0
        z[5, 0] = h[1, 0] * h[7, 0] + h[5, 0] * h[3, 0] + h[0, 0] * h[6, 0] + h[4, 0] * h[2, 0]
        z[6, 0] = h[2, 0] * h[7, 0] + h[6, 0] * h[3, 0] - h[0, 0] * h[5, 0] - h[4, 0] * h[1, 0]
        z[7, 0] = h[3, 0] * h[7, 0] - h[2, 0] * h[6, 0] - h[1, 0] * h[5, 0] + h[0, 0] * h[4, 0]

        j4 = dualQuatMult(z, xe)

        ##################################j5##################################
        h = dualQuatMult(hB_O6a, KinematicModel(MDH_right, thetar, 5, 1))

        z[0, 0] = 0
        z[1, 0] = h[1, 0] * h[3, 0] + h[0, 0] * h[2, 0]
        z[2, 0] = h[2, 0] * h[3, 0] - h[0, 0] * h[1, 0]
        z[3, 0] = (h[3, 0] ** 2 - h[2, 0] ** 2 - h[1, 0] ** 2 + h[0, 0] ** 2) / 2
        z[4, 0] = 0
        z[5, 0] = h[1, 0] * h[7, 0] + h[5, 0] * h[3, 0] + h[0, 0] * h[6, 0] + h[4, 0] * h[2, 0]
        z[6, 0] = h[2, 0] * h[7, 0] + h[6, 0] * h[3, 0] - h[0, 0] * h[5, 0] - h[4, 0] * h[1, 0]
        z[7, 0] = h[3, 0] * h[7, 0] - h[2, 0] * h[6, 0] - h[1, 0] * h[5, 0] + h[0, 0] * h[4, 0]

        j5 = dualQuatMult(z, xe)

    jac = np.concatenate((j0, j1, j2, j3, j4, j5), axis=1)

    return jac
