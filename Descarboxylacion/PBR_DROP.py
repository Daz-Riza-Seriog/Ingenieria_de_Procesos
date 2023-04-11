# Code made from Equipo 1 Ingenieria de Procesos
# 26 Marzo 2023
# License MIT
# Ingenieria de Procesos


import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
import seaborn as sns

from Parametros_Finales import *
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
        F_AO = x[0]  # molar flow rate Oleic Acid
        F_H = x[1]  # molar flow rate of H2
        F_A = x[2]  # molar flow rate of Stearic Acid
        F_B = x[3]  # molar flow rate of Alcohol Estearico
        F_C = x[4]  # molar flow rate of Heptadecano
        F_D = x[5]  # molar flow rate of Octadecano
        F_E = x[6]  # molar flow rate of Agua
        P = x[7]  # total pressure
        
        # compute total molar flow rate
        F_tot = F_AO + F_H + F_A + F_B + F_C + F_D + F_E 
        # compute volumetric flow rate in m^3/s
        R = Reactor.R  # Bar*m^3/mol*K
        T = Reactor.T_0  # isothermal operation [K]
        Z = 1  # Acentric Factor
        
        vflow = (F_tot / Reactor.F_tot0) * (P / Reactor.P_0) * (Reactor.T_0 / T) * Reactor.vflow_0  # volumetric flow rate
        
        # compute partial pressures in gas phase
        C_AO = F_AO/vflow # Concentracion de AO [mol/m^3]
        P_H = (F_H * Z * R * T) / vflow  # Presion parcial H2 [bar]
        C_A = F_A /vflow # Concentracion de Acido Estearico [mol/m^3]
        C_B = F_B /vflow # Concentracion de Alcohol Estearico [mol/m^3]
        
        # pressure drop across packed bed
        var_1 = Reactor.beta_0 / (Reactor.area * (1 - Reactor.f) * Reactor.rc)  # Poner en terminos de volumen
        dP_dW = (var_1 * (Reactor.P_0 / P) * (F_tot / Reactor.F_tot0))  # Poner en terminos de volumen
        
        # Compute reaction rate
        # Inhibition Parameter
        inh = (1 + (Rate.K_A * C_A) )*(1+(Rate.K_H*P_H))

        # Rate law 1
        r_1 = Rate.k_1*C_AO # [mol AO/min*g_cat]
        
        # Rate law 2
        r_2 = ((Rate.k_1_prima * C_A * P_H)/inh)/Reactor.rb #[mol A/min*g_cat]

        # Rate law 3
        r_3 = (Rate.k_2_prima * C_B / (1 + Rate.K_A * C_A))/Reactor.rb #[mol B/min*g_cat]

        # Rate law 4
        r_4 = (Rate.k_3_prima * C_B * P_H / inh)/Reactor.rb #[mol B/min*g_cat]

        # leyes de Velocidad Netas para cada componente
        r_AO = -r_1 
        r_H = -r_1 - (2 *r_2) - r_4
        r_A = r_1 - r_2
        r_B = r_2 - r_3 - r_4
        r_C = r_3
        r_D = r_4
        r_E = r_2 + r_4

        # mole balance on AO
        dFAO_dW = r_AO
        # mole balance on H2
        dFH_dW = r_H
        # mole balance on A
        dFA_dW = r_A
        # mole balance on B
        dFB_dW = r_B
        # mole balance on C
        dFC_dW = r_C
        # mole balance on D
        dFD_dW = r_D
        # mole balance on E
        dFE_dW = r_E
        
        f = [dFAO_dW, dFH_dW, dFA_dW,  dFB_dW, dFC_dW, dFD_dW, dFE_dW, dP_dW]

        return f


class PLOT:
    def __init__(self,Sol_ODE_PBR):
        self.Sol_ODE_PBR = Sol_ODE_PBR

    def plots(self):
        sol_PBR = self.Sol_ODE_PBR

        plt.figure(1)
        plt.title("PBR\nCatalizador vs Flujo")
        plt.plot(sol_PBR.t, sol_PBR.y[0], label=r"$F_{AO}$")
        plt.plot(sol_PBR.t, sol_PBR.y[1], label=r"$F_{H_2}$")
        plt.plot(sol_PBR.t, sol_PBR.y[2], label=r"$F_{A}$")
        plt.plot(sol_PBR.t, sol_PBR.y[3], label=r"$F_{B}$")
        plt.plot(sol_PBR.t, sol_PBR.y[4], label=r"$F_{C}$")
        plt.plot(sol_PBR.t, sol_PBR.y[5], label=r"$F_{D}$")
        plt.plot(sol_PBR.t, sol_PBR.y[6], label=r"$F_{E}$")
        plt.xlabel('Catalizador $[g]$', labelpad=15, fontsize=13)
        plt.ylabel('Flujo molar $F_i$ $[mol/min]$', labelpad=8, fontsize=12)
        plt.legend()
        plt.tight_layout()

        plt.figure(2)
        plt.title("PBR\nCaida de Presion vs Peso Catalizador")
        # Aca hay un machetazo normalizado en la caida de Presion
        plt.plot(sol_PBR.t, sol_PBR.y[7][0]+(sol_PBR.y[7]/(sol_PBR.y[7][0]-sol_PBR.y[7][-1])), label=r"$Drop$")
        plt.xlabel('Catalizador $[g]$', labelpad=15, fontsize=13)
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
