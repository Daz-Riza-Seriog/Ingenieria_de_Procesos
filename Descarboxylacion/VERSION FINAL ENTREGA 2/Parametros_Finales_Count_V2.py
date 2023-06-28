# Code made by Sergio Andrés Díaz Ariza
# 30 September 2022
# License MIT
# IRQ: Python Program-Assigment 1

import numpy as np

# coolprop libreria usar para propiedades termodinamicas

class set_Reactor:
    # Supose a Initial Volume of Reactor of PBR
    W_tot_su = 30000 # kg_cata
    
    ## VALORES INICIALES DADOS POR EL PAPER ##
    # Constant Ideal Gas for work in stoichiometry
    R = 8.3144626181532e-5  # Bar*m^3/K*mol
    # total pressure of inlet stream
    P_0 = 30  # in Bar initial
    # inlet temperature
    T_0 = 300 + 273.15  # in K
    # Temperatura inicial de Refrigerante
    Ta_0 = 300.65 + 273.15 # K

    # Flujo del refrigerante 
    F_cool = 0.32  # [kmol/min]

    # Flujo molar de Acido Oleico
    F_A0 = 5.77/60 # Valor dado en el trabajo anterior derivado de la meta a producir [kmol/min]
    # molar flow rate of H2 in feed stream
    F_H0 = 3 * F_A0 + 0.0923443628353128 # [kmol/min]
    F_B0 = 0
    F_C0 = 0
    F_D0 = 0
    F_E0 = 0
    F_CO0 = 0
    
    # total molar flow rate of feed stream in [kmol/min]
    F_tot0 = F_A0 + F_H0 + F_B0 + F_C0 + F_D0 + F_E0 + F_C0 
    # Fraccion molar del Acido Oleico Inicial
    y_A_0 = F_A0 / F_tot0
    y_H_0 = F_H0 / F_tot0
    y_B_0 = F_B0 / F_tot0
    y_C_0 = F_C0 / F_tot0
    y_D_0 = F_D0 / F_tot0
    y_E_0 = F_E0 / F_tot0
    y_CO_0 = F_CO0 / F_tot0

    # partial pressure of AO in inlet stream
    p_A0 = y_A_0 * P_0  # Bar
    # partial pressure of H in inlet stream
    p_H0 = y_H_0 * P_0  # Bar
    # Concentrations
    C_tot0 = (P_0/(R*T_0)) #[Kmol/m^3] -----> no es mol/m^3
    C_A0 = p_A0 / (R * T_0) #[Kmol/m^3]
    C_H0 = p_H0 / (R * T_0) #[Kmol/m^3]
    # Voumetric Initial Flow
    vflow_0 = F_tot0 / C_tot0  # [m^3/min]

    # Calculo de Capacidades Calorificas
    Cp_H = 29.349  #  del H2 en [kJ/kmol*K] 
    Cp_A = 223.628  #  del A en [kJ/kmol*K]
    Cp_B = 191.07  #  del B en [kJ/kmol*K]
    Cp_C = 180.755  #  C en [kJ/kmol*K]
    Cp_D = 189.081  #  D en [kJ/kmol*K]
    Cp_E = 39.6861  #  E en [kJ/kmol*K]
    Cp_CO = 30.5431 #  CO en [kJ/kmol*K]
    Cp_F = 51.3205 #  CO en [kJ/kmol*K]

    # Calculo de Gamma ESTO ES DEL TRABAJO ANTERIOR CON APROXIMACIONES
    PM_H = 0.00201588  # Peso molecular del H2 en [kg/mol]
    PM_A = 0.284483  # Peso molecular del A en [kg/mol]
    PM_B = 0.270499  # Peso molecular del B en [kg/mol]
    PM_C = 0.240473  # Peso molecular del C en [kg/mol]
    PM_D = 0.25449432  # Peso molecular del D en [kg/mol]
    PM_E = 0.0180153  # Peso molecular del E en [kg/mol]
    PM_CO = 0.0280104 # Peso molecular del CO en [kg/mol]
    PM_F = 0.0300263 # Peso molecular del Fornaldehyde en [kg/mol]
    F_MASS = PM_A * F_A0 + PM_H * F_H0  # Flujo másico [kg/min]
    

    # Aproximación de la viscosidad de mezcla inicial
    mu_H = 0.0123176*(0.001)*(60)  # [cP]->  kg/m*s -> kg/m*min
    mu_A = 0.40824*(0.001)*(60) # [cP]->  kg/m*s -> kg/m*min
    mu_B = 0.307132*(0.001)*(60) # [cP]->  kg/m*s -> kg/m*min
    mu_C = 0.218577*(0.001)*(60) # [cP]->  kg/m*s -> kg/m*min
    mu_D = 0.239512*(0.001)*(60) # [cP]->  kg/m*s -> kg/m*min
    mu_E = 0.0217708*(0.001)*(60) # [cP]->  kg/m*s -> kg/m*min
    mu_CO = 0.0271654*(0.001)*(60) # [cP]->  kg/m*s -> kg/m*min
    mu_F = 0.0323983*(0.001)*(60) # [cP]->  kg/m*s -> kg/m*min

    mu = (y_A_0 * mu_A) + (y_H_0 * mu_H)  # kg/m*min
    # Aproximación de la densidad del gas
    PM_prom = (y_A_0 * PM_A) + (y_H_0 * PM_H)  # Peso molecular promedio de la mezcla [kg/mol]
    ro = (P_0 * PM_prom) / (T_0 * R)  # kg/m^3 Usando gas ideal densidad inicial de mezcla
    
    ## VALORES PARA EL LECHO-!!

    # AREA DEL REACTOR
    di = 3.5  # Diametro interno [m]
    area = np.pi*((di**2)/4)  # Area transversal
    
    # Diametro de particula
    Dp = 0.0015  # m
    # Calculo de la porosidad
    f = 0.97  # Porosidad
    # rc = 3987  # [kg/m3] Densidad del catalizador 1
    rc = 944  # [kg/m3] Densidad del catalizador 2 
    rb = rc*(1-f) # [kg/m3] Densidad de bulto

    # Global Coefficient of Transfer Heat -TODO toca hallarlo esto es inventado
    Ua = 0.01  # [kJ/s*m^3*K]
    # Cp para refrigerante
    Cp_cool = 75.309  # [kJ/kmol*K] Agua a 30°C 



Reactor = set_Reactor()

class set_Rate:
    k_1_prima = 1.56e-3 #[1/min*bar]
    k_2_prima = 4.31-1 #[1/min]
    k_3_prima = 7.83e-4 #[1/min*bar]
    K_A = 1.36e2 #[m^3/kmol]
    K_H = 1.58e-2 #[1/bar]

    H_r1 = -40718  # kJ/kmol
    H_r2 = -675066 # kJ/kmol
    H_r3 = -94038  # kJ/kmol

class set_initial_state_flux_drop:
    __metaclass__ = Reactor
    x_0 = np.empty(8)  # allocate memory
    x_0[0] = Reactor.F_A0  # molar flow rate of A
    x_0[1] = Reactor.F_H0  # molar flow rate of H2
    x_0[2] = np.finfo(float).eps  # molar flow rate of B
    x_0[3] = np.finfo(float).eps  # molar flow rate of C
    x_0[4] = np.finfo(float).eps  # molar flow rate of D
    x_0[5] = np.finfo(float).eps  # molar flow rate of E
    x_0[6] = np.finfo(float).eps  # molar flow rate of CO
    x_0[7] = Reactor.P_0  # pressure

class set_initial_state_flux_drop_adiabatic:
    __metaclass__ = Reactor
    x_0 = np.empty(9)  # allocate memory
    x_0[0] = Reactor.F_A0  # molar flow rate of AO
    x_0[1] = Reactor.F_H0  # molar flow rate of H2
    x_0[2] = np.finfo(float).eps  # molar flow rate of B
    x_0[3] = np.finfo(float).eps  # molar flow rate of C
    x_0[4] = np.finfo(float).eps  # molar flow rate of D
    x_0[5] = np.finfo(float).eps  # molar flow rate of E
    x_0[6] = np.finfo(float).eps  # molar flow rate of CO
    x_0[7] = Reactor.P_0  # pressure
    x_0[8] = Reactor.T_0 # Temperature

class set_initial_state_flux_drop_jacket:
    __metaclass__ = Reactor
    x_0 = np.empty(10)  # allocate memory
    x_0[0] = Reactor.F_A0  # molar flow rate of AO
    x_0[1] = Reactor.F_H0  # molar flow rate of H2
    x_0[2] = np.finfo(float).eps  # molar flow rate of B
    x_0[3] = np.finfo(float).eps  # molar flow rate of C
    x_0[4] = np.finfo(float).eps  # molar flow rate of D
    x_0[5] = np.finfo(float).eps  # molar flow rate of E
    x_0[6] = np.finfo(float).eps  # molar flow rate of CO
    x_0[7] = Reactor.P_0  # pressure
    x_0[8] = Reactor.T_0 # Temperature
    x_0[9] = Reactor.Ta_0
