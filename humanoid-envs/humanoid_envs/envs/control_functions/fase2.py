import math as mt
import time
import warnings

import numpy as np

from humanoid_envs.envs.control_functions.dualHamiltonOp import dualHamiltonOp
from humanoid_envs.envs.control_functions.dualQuatConj import dualQuatConj
from humanoid_envs.envs.control_functions.dualQuatMult import dualQuatMult
from humanoid_envs.envs.control_functions.getPositionDualQuat import getPositionDualQuat
from humanoid_envs.envs.control_functions.globalVariables import GlobalVariables
from humanoid_envs.envs.control_functions.jacobianoCoM import jacobiano2
from humanoid_envs.envs.control_functions.jacobianoPes import jacobianoPes
from humanoid_envs.envs.control_functions.kinematicModel import KinematicModel
from humanoid_envs.envs.control_functions.kinematicRobo import kinematicRobo
from humanoid_envs.envs.control_functions.transformacao import transformacao

warnings.filterwarnings("error")


# -----------------------------------
# Método para executar o  passo com a perna esquerda como suporte da
# caminhada e a perna direita em movimento
# -----------------------------------
def fase2(trajCoM, ind, trajPA, theta, vecGanho, robot, simulator, client, joint_names, init_angles):
    glob = GlobalVariables()
    hEdo = glob.getHEDO()
    L1 = glob.getL1()
    MDH_left = glob.getMDH_left()
    hpi = glob.getHpi()

    dt = hEdo  # dt é o tempo da solução da equação Edo
    n = np.array([0, 1, 0])  # n é o vetor diretor do quaternio
    thetab = hpi  # parametro da função de caminhada que é igual a pi/2
    realRb = np.array([np.cos(thetab / 2)])
    hOrg = np.array([1, 0, 0, 0, 0, 0, 0, 0]).reshape((8, 1))  # posição da base
    rb = np.concatenate((realRb, np.sin(thetab / 2) * n), axis=0).reshape((4, 1))
    pb = np.array([[0], [0], [0], [0]])

    # base B para a base O6 (perna em movimento)
    hB_O6 = transformacao(pb, rb)
    hP = dualQuatMult(hOrg, hB_O6)
    T = np.size(trajCoM, 0)

    # matrizes auxiliares
    Mhd = np.zeros((8, T))
    Mdhd = np.zeros((8, T))

    Mhd2 = np.zeros((8, T))
    Mdhd2 = np.zeros((8, T))

    ha2 = kinematicRobo(theta, hOrg, hP, 0, 0)
    ha = kinematicRobo(theta, hOrg, hP, 0, 1)
    pos_CoM = getPositionDualQuat(ha)
    pos_foot = getPositionDualQuat(ha2)
    trajCoM[:, 0] = trajCoM[:, 0] + pos_CoM[0, 0]
    trajCoM[:, 1] = trajCoM[:, 1] - L1  # 2*pos_CoM[1,0]

    trajPA[:, 0] = trajPA[:, 0] + pos_foot[0, 0]
    trajPA[:, 1] = trajPA[:, 1] - L1 * 2
    trajPA[:, 2] = trajPA[:, 2]  # + pos_foot[2,0]

    # calculo de Mhd - matriz de hd
    r = np.array([1, 0, 0, 0]).reshape((4, 1))
    p = np.array([0, 0, 0, 0]).reshape((4, 1))
    hB1 = transformacao(p, r)  # transformação base robô
    for i in range(0, T, 1):
        p = np.array([0, trajCoM[i, 0], trajCoM[i, 1], trajCoM[i, 2]]).reshape((4, 1))
        r = np.array([1, 0, 0, 0]).reshape((4, 1))
        hd = transformacao(p, r)  # posição desejada do CoM
        mhd = dualQuatMult(hB1, hd)
        Mhd[:, i] = mhd[:, 0]

        if i < ind:
            p = np.array([0, trajPA[i, 0], trajPA[i, 1], trajPA[i, 2]]).reshape((4, 1))
            n = np.array([0, 1, 0])
            angulo = mt.pi / 2
            realR = np.array([mt.cos(angulo / 2)])
            imagR = mt.sin(angulo / 2) * n
            rb = np.concatenate((realR, imagR), axis=0).reshape((4, 1))
            hd = transformacao(p, rb)  # posição desejada do pé
            mhd2 = dualQuatMult(hB1, hd)
            Mhd2[:, i] = mhd2[:, 0]  # transformação da base até o pé
        else:
            Mhd2[:, i] = Mhd2[:, ind - 1]

    for i in range(1, T, 1):
        Mdhd[:, i] = (Mhd[:, i] - Mhd[:, i - 1]) * (1 / dt)
        Mdhd2[:, i] = (Mhd2[:, i] - Mhd2[:, i - 1]) * (1 / dt)

    # LQR
    ganhoS = vecGanho[0, 0]
    ganhoQ = vecGanho[1, 0]
    ganhoR = vecGanho[2, 0]

    # controlador proporcional
    ganhoK2 = vecGanho[3, 0]
    K2 = ganhoK2 * np.eye(8)

    # ganho P-FF
    S = ganhoS * np.eye(8)
    Q = ganhoQ * np.eye(8)
    R = ganhoR * np.eye(8)
    Rinv = np.linalg.inv(R)
    C8 = np.diag([1, -1, -1, -1, 1, -1, -1, -1])

    # iniciar condições finais esperadas para P e E
    Pf = S
    Ef = np.array([0, 0, 0, 0, 0, 0, 0, 0]).reshape((8, 1))
    P = Pf
    MP2 = np.zeros((8, 8, T))
    for j in range(8):
        for k in range(8):
            MP2[j, k, T - 1] = P[j, k]
    E = Ef
    ME2 = np.zeros((8, T))
    ME2[:, T - 1] = E[:, 0]

    for i in range(T - 2, -1, -1):
        aux = dualQuatMult(dualQuatConj(Mhd[:, i + 1].reshape((8, 1))), Mdhd[:, i + 1].reshape((8, 1)))
        A = dualHamiltonOp(aux, 0)
        c = -aux
        try:
            P = P - (-P @ A - A.T @ P + P @ Rinv @ P - Q) * dt
            for j in range(8):
                for k in range(8):
                    MP2[j, k, i] = P[j, k]
            E = E - (-1 * (A.T) @ E + P @ Rinv @ E - P @ c) * dt
            ME2[:, i] = E[:, 0]
        except RuntimeWarning:
            return ha, ha2, theta, np.zeros((3, 1)), np.zeros((3, 1)), -1

    t0 = time.time()
    for i in range(0, T, 1):
        # Controlador LQR para O CoM
        # calculo de A e c
        aux = dualQuatMult(dualQuatConj(Mhd[:, i].reshape((8, 1))), Mdhd[:, i].reshape((8, 1)))
        # inicio do controlador
        xe = KinematicModel(MDH_left, theta, 6, 0)
        Ja = jacobiano2(theta, hOrg, hP, xe, 1)

        # calculo de P e E
        # calculo de N
        Hd = dualHamiltonOp(Mhd[:, i].reshape((8, 1)), 0)
        # prod3 = np.dot(Hd,C8)
        N = Hd @ C8 @ Ja
        # pseudo inversa de N
        Np = np.linalg.pinv(N)
        # calculo do erro
        e = np.array([1, 0, 0, 0, 0, 0, 0, 0]).reshape((8, 1)) - dualQuatMult(dualQuatConj(ha),
                                                                              Mhd[:, i].reshape((8, 1)))
        # calculo de P e E
        E[:, 0] = ME2[:, i]
        P[:, :] = MP2[:, :, i].reshape((8, 8))
        do = Np @ Rinv @ (P @ e + E)
        # calculo do o deseja
        od = dt * do / 2
        theta[:, 1] = theta[:, 1] + od[:, 0]

        for j in range(0, 6, 1):
            if abs(theta[j, 1]) > hpi:
                theta[j, 1] = np.sign(theta[j, 1]) * hpi

        ha = kinematicRobo(theta, hOrg, hP, 0, 1)  # posição do CoM com perna esquerda apoiada

        # controlador 2
        # calculo de A e c
        aux2 = dualQuatMult(dualQuatConj(Mhd2[:, i].reshape((8, 1))), Mdhd2[:, i].reshape((8, 1)))
        # inicio do controlador
        xe2 = kinematicRobo(theta, hOrg, hP, 1, 0)
        Ja2 = jacobianoPes(theta, ha, xe2, 0)
        # calculo de P e E
        # calculo de N
        Hd2 = dualHamiltonOp(Mhd2[:, i].reshape((8, 1)), 0)
        N2 = Hd2 @ C8 @ Ja2

        # pseudo inversa de N
        Np2 = np.linalg.pinv(N2)

        # calculo do erro
        e2 = np.array([1, 0, 0, 0, 0, 0, 0, 0]).reshape((8, 1)) - dualQuatMult(dualQuatConj(ha2),
                                                                               Mhd2[:, i].reshape((8, 1)))

        vec2 = dualQuatMult(dualQuatConj(ha2), Mdhd2[:, i].reshape((8, 1)))
        do2 = Np2 @ (K2 @ e2 - vec2)
        od2 = (do2 * dt) / 2
        theta[:, 0] = theta[:, 0] + od2[:, 0]
        if abs(theta[j, 0]) > hpi:
            theta[j, 0] = np.sign(theta[j, 0]) * hpi

        # send the angles to the simulator
        joints = init_angles
        joints[10:16] = theta[:, 1]  # left leg
        joints[16:22] = theta[:, 0]  # right leg
        # Inverte 'LHipYawPitch' e 'RHipYawPitch'
        joints[10] = -joints[10]
        joints[16] = -joints[16]

        robot.setAngles(joint_names, joints, 1)
        simulator.stepSimulation(client)

        if robot.getLinkPosition("Head")[0][2] < 0.15:
            return ha, ha2, theta, np.zeros((3, 1)), np.zeros((3, 1)), -1

        # get the real angles to calculate the actual position
        angles = robot.getAnglesPosition(joint_names)

        # Inverts 'LHipYawPitch' e 'RHipYawPitch'
        angles[10] = -angles[10]
        angles[16] = -angles[16]

        theta[:, 1] = angles[10:16]
        theta[:, 0] = angles[16:22]

        ha = kinematicRobo(theta, hOrg, hP, 0, 1)
        ha2 = kinematicRobo(theta, hOrg, hP, 0, 0)
    t1 = time.time()
    t = t1 - t0

    pos_CoM = getPositionDualQuat(ha)  # CoM position with the left leg supporting
    pos_leg = getPositionDualQuat(ha2)  # right leg position, as swinging leg

    return ha, ha2, theta, pos_CoM, pos_leg, t
