import numpy as np


class GlobalVariables:
    thetaR = np.zeros((6, 1))
    thetaL = np.zeros((6, 1))
    hpi = (np.pi) / 2.0

    # marta

    L1 = 0.05
    L2 = 0.085
    L3 = 0.100
    L4 = 0.10274
    L5 = 0.04511

    height = L2 + L3 + L4 + L5

    # tabela DH 1
    oi = np.array([0, -np.pi / 4, 0.0, 0.0, 0.0, 0]).reshape((6, 1))
    di = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape((6, 1))
    ai = np.array([0.0, 0.0, L3, L4, 0, L5]).reshape((6, 1))
    si = np.array([hpi, -hpi, 0.0, 0.0, hpi, 0]).reshape((6, 1))

    oi_l = np.array([0, -3 * np.pi / 4, 0.0, 0.0, 0.0, 0.0]).reshape((6, 1))
    di_l = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape((6, 1))
    ai_l = np.array([0.0, 0.0, L3, L4, 0, L5]).reshape((6, 1))
    si_l = np.array([hpi, -hpi, 0.0, 0.0, hpi, 0]).reshape((6, 1))

    ori = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape((6, 1))  # perna direita
    ol = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape((6, 1))  # perna esquerda
    MDH_right = np.zeros((6, 4))
    for j in range(6):
        MDH_right[j, 0] = oi[j, 0]
        MDH_right[j, 1] = di[j, 0]
        MDH_right[j, 2] = ai[j, 0]
        MDH_right[j, 3] = si[j, 0]

    MDH_left = np.zeros((6, 4))
    for j in range(6):
        MDH_left[j, 0] = oi_l[j, 0]
        MDH_left[j, 1] = di_l[j, 0]
        MDH_left[j, 2] = ai_l[j, 0]
        MDH_left[j, 3] = si_l[j, 0]

    hEdo = 10 ** (-3)  # passo para o cálculo da EDO
    m = 58.2  # HRP
    L = 0.204
    ts = 0.3  # ADJUST
    tdbl = 0.05  # adjust
    g = 9.8
    h = 10 ** (-5)  # passo para o calculo das derivadas
    maxNGrad = 10 ** 6  # número máximo de iterações método
    ganhoAlpha = 10 ** (-1)  # ganho do fator de ganho para cada passo
    gamma = 0.2  # ganho para os método gradiente(momento)
    thetaM = 0.5
    phiM = 0.5
    KM = 2.0
    BSSM = 0.2
    pfa = np.array([0.0, 0.0, 0.0]).reshape((3, 1))  # posição do pé de suporte em MS
    expK = 1000.0  # ordem de grandeza da constante massa-mola Darwin

    # DArwin dynamic model parameters
    phi = 0.6482570031
    theta = 0.2823507428
    k = 153.9300628927
    Bss = 0.0414743461

    def getHpi(self):
        return self.hpi

    def getL1(self):
        return self.L1

    def getL2(self):
        return self.L2

    def getL3(self):
        return self.L3

    def getL4(self):
        return self.L4

    def getL5(self):
        return self.L5

    def getHeight(self):
        return self.height

    def getMDH_left(self):
        return self.MDH_left

    def getMDH_right(self):
        return self.MDH_right

    def getHEDO(self):
        return self.hEdo

    def getH(self):
        return self.h

    def getMaxNGrad(self):
        return self.maxNGrad

    def getGanhoAlpha(self):
        return self.ganhoAlpha

    def getGamma(self):
        return self.gamma

    def getThetaM(self):
        return self.thetaM

    def getPhiM(self):
        return self.phiM

    def getKM(self):
        return self.KM

    def getBSSM(self):
        return self.BSSM

    def getPfa(self):
        return self.pfa

    def getExpK(self):
        return self.expK

    def getM(self):
        return self.m

    def getL(self):
        return self.L

    def getG(self):
        return self.g

    def getOr(self):
        return self.ori

    def getOl(self):
        return self.ol

    def getPhi(self):
        return self.phi

    def getTheta(self):
        return self.theta

    def getK(self):
        return self.k

    def getBss(self):
        return self.Bss

    def setPhi(self, phi):
        self.phi = phi

    def setTheta(self, theta):
        self.theta = theta

    def setK(self, k):
        self.k = k

    def setBss(self, Bss):
        self.Bss = Bss

    def setThetaR(self, thetaR):
        self.thetaR = thetaR

    def setThetaL(self, thetaL):
        self.thetaL = thetaL

    def getThetaR(self):
        return self.thetaR

    def getThetaL(self):
        return self.thetaL
