# Code made from Equipo 1 Ingenieria de Procesos
# 26 Marzo 2023
# License MIT
# Ingenieria de Procesos


import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
import seaborn as sns

from Parametros_Finales_V2 import *
import timeit

start = timeit.default_timer()
sns.set()

ini_state = set_initial_state_flux_drop()
Rate = set_Rate()

class ODE_PBR:

    def __init__(self,x,Rate,Reactor) -> None:
        self.x = x
        self.Rate = Rate
        self.Reactor = Reactor

    # PBR_calc_f function
    def dFdW(self,W,x):
        # extract state variables
        F_A = x[0]  # molar flow rate Stearic Acid
        F_H = x[1]  # molar flow rate of H2
        F_B = x[2]  # molar flow rate of Alcohol Estearico
        F_C = x[3]  # molar flow rate of Octadequeno
        F_D = x[4]  # molar flow rate of Octadecano
        F_E = x[5]  # molar flow rate of Agua
        F_CO = x[6] # molar flow rate of CO
        P = x[7]  # total pressure
        
        # compute total molar flow rate
        F_tot = F_A + F_H  + F_B + F_C + F_D + F_E + F_CO # [kmol/min]
        # compute volumetric flow rate in m^3/s
        R = Reactor.R  # Bar*m^3/mol*K
        T = Reactor.T_0  # isothermal operation [K]
        Z = 1  # Acentric Factor

        # Fraccion molar de Componentes
        y_A = F_A / F_tot # Acido Estearico
        y_H = F_H / F_tot # Acido Hidrogeno
        y_B = F_B / F_tot # Estearil Alcohol
        y_C = F_C / F_tot # Heptadecano
        y_D = F_D / F_tot # Octadecano
        y_E = F_E / F_tot # Agua
        y_CO = F_CO / F_tot # CO
        
        #vflow = (F_tot / Reactor.F_tot0) * (P / Reactor.P_0) * (Reactor.T_0 / T) * Reactor.vflow_0  # volumetric flow rate [m3/min]
        CT_0 = (Reactor.P_0/(Reactor.R*Reactor.T_0))/1000 #kmol/m^3

        # compute partial pressures in gas phase
        C_A = CT_0 *(F_A/F_tot) *(P/Reactor.P_0)# Concentracion de AO [kmol/m^3]
        P_H = y_H*P  # Presion parcial H2 [bar]
        C_B = CT_0 *(F_B/F_tot) *(P/Reactor.P_0) # Concentracion de Alcohol Estearico [kmol/m^3]
        C_C = CT_0 *(F_C/F_tot) *(P/Reactor.P_0) # Concentracion de Heptadecano [kmol/m^3]
        C_D = CT_0 *(F_D/F_tot) *(P/Reactor.P_0) # Concentracion de Octadecano [kmol/m^3]
        C_E = CT_0 *(F_E/F_tot) *(P/Reactor.P_0) # Concentracion de Agua [kmol/m^3]
        C_CO = CT_0 *(F_CO/F_tot) *(P/Reactor.P_0) # Concentracion de CO [kmol/m^3]
        
        
        # pressure drop across packed bed
        flujo_m_tot = (F_A * Reactor.PM_A + F_H * Reactor.PM_H + F_B * Reactor.PM_B +
                        F_C * Reactor.PM_C + F_D * Reactor.PM_D + F_E * Reactor.PM_E +
                          F_CO * Reactor.PM_CO)*1000  # [kmol/min]*[kg/mol]*1000->[kg/min]
        
        # mu de mix variable
        mu = (y_A * Reactor.mu_A) + (y_H * Reactor.mu_H) +\
              (y_B * Reactor.mu_B) + (y_C * Reactor.mu_C) + (y_D * Reactor.mu_D) +\
                  (y_E * Reactor.mu_E) + (y_CO * Reactor.mu_CO) # kg/m*min
        
        # Aproximación de la densidad de mezcla variable
        PM_prom = (y_A * Reactor.PM_A) + (y_H * Reactor.PM_H) +\
              (y_B * Reactor.PM_B) + (y_C * Reactor.PM_C) + (y_D * Reactor.PM_D) +\
                  (y_E * Reactor.PM_E) + (y_CO * Reactor.PM_CO)# Peso molecular promedio de la mezcla [kg/mol]
        ro = (P * PM_prom) / (T * R)  # kg/m^3

        # densidad fija inicial
        #PM_prom = Reactor.y_A_0*Reactor.PM_A + Reactor.y_H_0*Reactor.PM_H
        #ro = (Reactor.P_0*PM_prom)/(R*Reactor.T_0)


        G = flujo_m_tot / Reactor.area  # [kg/m^2 * min]
        ter1 = ((150 * (1 - Reactor.f) * mu) / Reactor.Dp) # [kg/m^2*min]
        ter2 = (G * (1 - Reactor.f)) / (ro * Reactor.Dp * (Reactor.f ** 3)) # [1/min]
        beta_0 = ter2 * (ter1 + (1.75 * G))*1e-5*1/60**2 #  [bar/m]
        alpha = (2*beta_0)/((Reactor.area*Reactor.rc*(1-Reactor.f))*Reactor.P_0)                                              #[1/kg]
        dP_dW = -(alpha/2)*((Reactor.P_0**2)/P)*(F_tot/Reactor.F_tot0)

        # kg/m^2*min^2----> kg*m/s^2 = N y 1/60^2 m/m ---- > kg/m^2*s^2 m/m ----- N/m^2 * 1/m------ N/m^2 = 1e5 bar/m 

        # Compute reaction rate
        # Inhibition Parameter
        inh = (1 + (Rate.K_A * C_A) )*(1+(Rate.K_H*P_H))

        # Rate law 1
        r_1 = (((Rate.k_1_prima * C_A * P_H)/inh))/1000 # [kmol/L*min]

        # Rate law 2
        r_2 = ((Rate.k_2_prima * C_B / (1 + Rate.K_A * C_A)))/1000 #[kmol/L*min]

        # Rate law 3
        r_3 = ((Rate.k_3_prima * C_B * P_H / inh))/1000 #[kmol/L*min]

        # leyes de Velocidad Netas para cada componente
        r_A = -r_1 
        r_H = (-2*r_1) - r_2 -r_3 
        r_B = r_1 - (34*r_2) -r_3
        r_C = 35*r_2
        r_D = r_3
        r_E = r_1 + (17*r_2) +r_3
        r_CO = 17*r_2

        # mole balance on A
        dFA_dW = r_A*(1/Reactor.rb)*1000 #[kmol/min*Kg_cat]
        # mole balance on H2
        dFH_dW = r_H*(1/Reactor.rb)*1000 #[kmol/min*Kg_cat]
        # mole balance on B
        dFB_dW = r_B*(1/Reactor.rb)*1000 #[kmol/min*Kg_cat]
        # mole balance on C
        dFC_dW = r_C*(1/Reactor.rb)*1000 #[kmol/min*Kg_cat]
        # mole balance on D
        dFD_dW = r_D*(1/Reactor.rb)*1000 #[kmol/min*Kg_cat]
        # mole balance on E
        dFE_dW = r_E*(1/Reactor.rb)*1000 #[kmol/min*Kg_cat]
        # mole balance on E
        dFCO_dW = r_CO*(1/Reactor.rb)*1000 #[kmol/min*Kg_cat]
        
        f = [dFA_dW, dFH_dW, dFB_dW, dFC_dW, dFD_dW, dFE_dW,dFCO_dW, dP_dW]

        return f


class PLOT:
    def __init__(self,Sol_ODE_PBR):
        self.Sol_ODE_PBR = Sol_ODE_PBR

    def plots(self):
        sol_PBR = self.Sol_ODE_PBR

        plt.figure(2)
        plt.title("PBR\nCatalizador vs Flujo")
        plt.plot(sol_PBR.t, sol_PBR.y[0], label=r"$F_{A}$")
        plt.plot(sol_PBR.t, sol_PBR.y[1], label=r"$F_{H_2}$")
        plt.plot(sol_PBR.t, sol_PBR.y[2], label=r"$F_{B}$")
        plt.plot(sol_PBR.t, sol_PBR.y[3], label=r"$F_{C}$")
        plt.plot(sol_PBR.t, sol_PBR.y[4], label=r"$F_{D}$")
        plt.plot(sol_PBR.t, sol_PBR.y[5], label=r"$F_{E}$")
        plt.plot(sol_PBR.t, sol_PBR.y[6], label=r"$F_{F}$")
        plt.xlabel('Catalizador $[kg]$', labelpad=15, fontsize=13)
        plt.ylabel('Flujo molar $F_i$ $[kmol/min]$', labelpad=8, fontsize=12)
        plt.legend()
        plt.tight_layout()

        plt.figure(3)
        plt.title("PBR\nCaida de Presion vs Peso Catalizador")
        plt.plot(sol_PBR.t, sol_PBR.y[7], label=r"$Drop$")
        plt.xlabel('Catalizador $[kg]$', labelpad=15, fontsize=13)
        plt.ylabel('Flujo molar $P$ $[Bar]$', labelpad=8, fontsize=12)
        plt.legend()
        plt.tight_layout()
        
        plt.show()

ODE_1 = ODE_PBR(ini_state.x_0,Rate,Reactor)

t_eval = np.linspace(0, Reactor.W_tot_su, 10000)
sol_PBR_1 = solve_ivp(ODE_1.dFdW,[0, Reactor.W_tot_su], ini_state.x_0, t_eval=t_eval, rtol=1e-12,
                    atol=1e-12)

Graph_PLOT_1 = PLOT(sol_PBR_1)
Graph_PLOT_1.plots()


stop = timeit.default_timer()
print('Time: ', stop - start)
