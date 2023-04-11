# Code made by Sergio Andrés Díaz Ariza
# 30 September 2022
# License MIT
# IRQ: Python Program-Assigment 1

import numpy as np


class set_Reactor:
    # Supose a Initial Volume of Reactor of PBR
    W_tot_su = 30 # g_cata
    
    ## VALORES INICIALES DADOS POR EL PAPER ##
    # Constant Ideal Gas for work in stoichiometry
    R = 8.3144626181532e-5  # Bar*m^3/K*mol
    # total pressure of inlet stream
    P_0 = 30  # in Bar initial
    # inlet temperature
    T_0 = 300 + 273.15  # in K

    # Flujo molar de Acido Oleico
    F_AO0 = 2.05454*(1000/60) # Valor dado en el trabajo anterior derivado de la meta a producir [mol/min]
    # molar flow rate of H2 in feed stream
    F_H0 = 4 * F_AO0  # [mol/min]
    
    
    # total molar flow rate of feed stream in [mol/min]
    F_tot0 = F_AO0 + F_H0 
    # Fraccion molar del Acido Oleico Inicial
    y_AO_0 = F_AO0 / F_tot0
    # Molar Fraction in the inlet of H2
    y_H_0 = 1 - y_AO_0
    # partial pressure of AO in inlet stream
    p_AO0 = y_AO_0 * P_0  # Bar
    # partial pressure of H in inlet stream
    p_H0 = y_H_0 * P_0  # Bar
    # Concentrations
    C_AO0 = p_AO0 / (R * T_0)
    C_H0 = p_H0 / (R * T_0)
    # Voumetric Initial Flow
    vflow_0 = F_AO0 / C_AO0  # [m^3/min]

    ## VALORES PARA EL LECHO-!!

    # AREA DEL REACTOR
    di = 0.70  # Diametro interno [m]
    area = np.pi * (di ** 2) / 4  # Area de los tubos 

    # Calculo de Gamma ESTO ES DEL TRABAJO ANTERIOR CON APROXIMACIONES
    PM_AO = 0.28246  # Peso molecular del AO en [kg/mol]
    PM_H = 0.002016  # Peso molecular del H2 en [kg/mol]
    F_MASS = PM_AO * F_AO0 + PM_H * F_H0  # Flujo másico [kg/min]
    gamma = F_MASS / area  # kg/min*m2
    
    # Aproximación de la viscosidad del gas
    mu_AO = 0.00198  # kg/m*min 
    mu_H = 0.0504  # kg/m*min
    mu = (y_AO_0 * mu_AO) + (y_H_0 * mu_H)  # kg/m*min
    # Aproximación de la densidad del gas
    PM_prom = (y_AO_0 * PM_AO) + (y_H_0 * PM_H)  # Peso molecular promedio de la mezcla [kg/mol]
    ro = (P_0 * PM_prom) / (T_0 * R)  # kg/m^3 Usando gas ideal
    # Diametro de particula
    Dp = 0.0015  # m
    # Calculo de la porosidad
    f = 0.78  # Correlación para hallar la porosidad
    rc = 944.3/1000  # g/m3 Densidad del catalizador 
    rb = rc*(1-f)

    # Ergun constant beta_0 for model of pressure
    # drop across packed bed, assumed constant viscosity
    var1 = (150 * (1 - f) * mu) / Dp
    var2 = (gamma * (1 - f)) / (ro * Dp * (f ** 3))  # Here the Value from matlab is different var2=34.7669
    beta_0 = var2 * (var1 + 1.75 * gamma)

Reactor = set_Reactor()

class set_Rate:
  k_1 = 10.9e-4 #[m^3/min*g_cat*mol_H2] !!OJO perro con el hidrogeno no se cancela, por tanto no se incluye
  # Disminui la ley de velocidad por que o sino es demasiado dominante respecto a las otras
  k_1_prima = 1.56e-3 #[1/min*bar]
  k_2_prima = 4.31e-1 #[1/min]
  k_3_prima = 7.83e-4 #[1/min*bar]
  K_A = 1.36e2/1000 #[m^3/mol]
  K_H = 1.58e-2 #[1/bar]                                          


class set_initial_state_flux_drop:
    __metaclass__ = Reactor
    x_0 = np.empty(8)  # allocate memory
    x_0[0] = Reactor.F_AO0  # molar flow rate of AO
    x_0[1] = Reactor.F_H0  # molar flow rate of H2
    x_0[2] = np.finfo(float).eps  # molar flow rate of A
    x_0[3] = np.finfo(float).eps  # molar flow rate of B
    x_0[4] = np.finfo(float).eps  # molar flow rate of C
    x_0[5] = np.finfo(float).eps  # molar flow rate of D
    x_0[6] = np.finfo(float).eps  # molar flow rate of E
    x_0[7] = Reactor.P_0  # pressure
