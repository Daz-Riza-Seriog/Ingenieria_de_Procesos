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

ini_state = set_initial_state_flux_drop_jacket()
ini_state2 = set_initial_state_flux_drop_jacket_recicle()
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
        F_C = x[3]  # molar flow rate of Heptadecano
        F_D = x[4]  # molar flow rate of Octadecano
        F_E = x[5]  # molar flow rate of Agua
        F_CO = x[6] # molar flow rate of CO
        P = x[7]  # total pressure
        T = x[8] # Temperature Reactor
        Ta = x[9] # Temeperatura Coolant
        
        
        # compute total molar flow rate
        F_tot = F_A + F_H  + F_B + F_C + F_D + F_E + F_CO # [kmol/min]
        # compute volumetric flow rate in m^3/s
        R = Reactor.R  # Bar*m^3/mol*K
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
                          F_CO * Reactor.PM_CO)*1000  # [kmol/min]*[kg/mol]*100->[kg/min]
        
        # mu de mix variable
        mu = (y_A * Reactor.mu_A) + (y_H * Reactor.mu_H) +\
              (y_B * Reactor.mu_B) + (y_C * Reactor.mu_C) + (y_D * Reactor.mu_D) +\
                  (y_E * Reactor.mu_E) + (y_CO * Reactor.mu_F) # kg/m*min
        
        # Aproximación de la densidad de mezcla variable
        PM_prom = (y_A * Reactor.PM_A) + (y_H * Reactor.PM_H) +\
              (y_B * Reactor.PM_B) + (y_C * Reactor.PM_C) + (y_D * Reactor.PM_D) +\
                  (y_E * Reactor.PM_E) + (y_CO * Reactor.PM_CO)# Peso molecular promedio de la mezcla [kg/mol]
        ro = (P * PM_prom) / (T * R)  # kg/m^3

        # densidad fija inicial
        #PM_prom = Reactor.y_A_0*Reactor.PM_A + Reactor.y_H_0*Reactor.PM_H
        #ro = (Reactor.P_0*PM_prom)/(R*Reactor.T_0)


        G = flujo_m_tot / Reactor.area  # [kg/m^2 * min]
        ter1 = ((150 * (1 - Reactor.f) * mu) / Reactor.Dp)
        ter2 = (G * (1 - Reactor.f)) / (ro * Reactor.Dp * (Reactor.f ** 3))
        beta_0 = ter2 * (ter1 + (1.75 * G))*1e-5*1/60**2 #  [bar/m]
        alpha = (2*beta_0)/((Reactor.area*Reactor.rc*(1-Reactor.f))*Reactor.P_0)   #[1/kg]
        dP_dW = -(alpha/2)*((Reactor.P_0**2)/P)*(F_tot/Reactor.F_tot0)

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

        # Adiabatico
        Qgen = (-r_1 * (Rate.H_r1)) + (-r_2 * (Rate.H_r2)) + (-r_3 * (Rate.H_r3))  #[kJ/kmol]
        Qrem = (Reactor.Ua /Reactor.rb) * (T - Ta)
        Cpmix = (F_A * Reactor.Cp_A) + (F_H * Reactor.Cp_H) +\
              (F_B * Reactor.Cp_B) + (F_C * Reactor.Cp_C) + (F_D * Reactor.Cp_D) +\
                  (F_E * Reactor.Cp_E) + (F_CO * Reactor.Cp_CO)
        dTadW = (Reactor.Ua) * (T - Ta) / (
                Reactor.F_cool * Reactor.Cp_cool)  # Co-Current Balance
        dT_dW = (Qgen - Qrem) / Cpmix

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
        
        f = [dFA_dW, dFH_dW, dFB_dW, dFC_dW, dFD_dW, dFE_dW, dFCO_dW, dP_dW, dT_dW, dTadW]

        return f


class PLOT:
    def __init__(self,Sol_ODE_PBR):
        self.Sol_ODE_PBR = Sol_ODE_PBR

    def plots(self):
        sol_PBR = self.Sol_ODE_PBR

        plt.figure(1)
        plt.title("PBR ADIABATICO\nCatalizador vs Flujo")
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

        plt.figure(2)
        plt.title("PBR ADIABTICO\nConversion de Reactivos")
        plt.plot(sol_PBR.t, 1-(sol_PBR.y[0]/sol_PBR.y[0][0]), label=r"$X_{AS}$")
        plt.plot(sol_PBR.t, 1-(sol_PBR.y[1]/sol_PBR.y[1][0]), label=r"$X_{H_2}$")
        plt.xlabel('Catalizador $[kg]$', labelpad=15, fontsize=13)
        plt.ylabel('Conversion $[-]$', labelpad=8, fontsize=12)
        plt.legend()
        plt.tight_layout()

        plt.figure(3)
        plt.title("PBR ADIABÁTICO\nCaída de Presión vs Peso Catalizador")
        plt.plot(sol_PBR.t, sol_PBR.y[7], label=r"$Drop$")
        plt.xlabel('Catalizador $[kg]$', labelpad=15, fontsize=13)
        plt.ylabel('Flujo molar $P$ $[Bar]$', labelpad=8, fontsize=12)
        plt.legend()
        plt.tight_layout()

        plt.figure(4)
        plt.title("PBR ADIABÁTICO\nCaída de Presión [%]")
        plt.plot(sol_PBR.t,( (sol_PBR.y[7][0]-sol_PBR.y[7])/sol_PBR.y[7][0])*100, label=r"$Drop$")
        plt.xlabel('Catalizador $[kg]$', labelpad=15, fontsize=13)
        plt.ylabel('Cambio en Presion $[Bar]$', labelpad=8, fontsize=12)
        plt.legend()
        plt.tight_layout()

        plt.figure(5)
        plt.title("PBR ADIABATICO\nTemperatua vs Peso Catalizador")
        plt.plot(sol_PBR.t, sol_PBR.y[8], label=r"$Temperature$") 
        plt.xlabel('Catalizador $[kg]$', labelpad=15, fontsize=13)
        plt.ylabel('Temperatura  $[K]$', labelpad=8, fontsize=12)
        plt.legend()
        plt.tight_layout()

        fig, axs = plt.subplots(2, 1)

        axs[0].set_title("PBR ADIABATICO\nTemperatura vs Peso Catalizador")
        axs[0].plot(sol_PBR.t, sol_PBR.y[8], label=r"Temperatura Reaactor")
        axs[0].set_ylabel('Temperatura  $[K]$', labelpad=8, fontsize=12)
        axs[0].legend()

        axs[1].plot(sol_PBR.t, sol_PBR.y[9], label=r"Temperatura Chaqueta", color="red")
        axs[1].set_xlabel('Catalizador $[kg]$', labelpad=15, fontsize=13)
        axs[1].set_ylabel('Temperatura  $[K]$', labelpad=8, fontsize=12)
        axs[1].legend()

        plt.tight_layout()
        
        plt.show()

ODE_1 = ODE_PBR(ini_state.x_0,Rate,Reactor)
ODE_2_recicle = ODE_PBR(ini_state2.x_0,Rate,Reactor)

t_eval = np.linspace(0, Reactor.W_tot_su, 10000)
sol_PBR_1 = solve_ivp(ODE_1.dFdW,[0, Reactor.W_tot_su], ini_state.x_0, t_eval=t_eval, rtol=1e-12,
                    atol=1e-12)

t_eval2 = np.linspace(0, Reactor.W_tot_su, 10000)
sol_PBR_2 = solve_ivp(ODE_2_recicle.dFdW,[0, Reactor.W_tot_su], ini_state2.x_0, t_eval=t_eval, rtol=1e-12,
                    atol=1e-12)

Graph_PLOT_1 = PLOT(sol_PBR_1)
Graph_PLOT_1.plots()

Graph_PLOT_2 = PLOT(sol_PBR_2)
Graph_PLOT_2.plots()





"""RESULTADO Y DATOS -> Put in the interactive"""

# # CONVERSIONES DE LOS REACTIVOS
# print("Conversion Acido Estearico\n:",100-(sol_PBR_1.y[0][-1]/sol_PBR_1.y[0][0])*100)
# print("Conversion Hidrogeno\n:",100-(sol_PBR_1.y[1][-1]/sol_PBR_1.y[1][0])*100)

# #FLUJOS MOLARES DE CADA ESPECIE A LA ENTRADA
# print("\nFlujo molar a la entrada de Acido Estearico [kmol/h]\n:",sol_PBR_1.y[0][0]*60) #kmol/h
# print("\nFlujo molar a la entrada de Hidrogeno [kmol/h]\n:",sol_PBR_1.y[1][0]*60)
# print("\nFlujo molar a la entrada de Estearil Alcohol [kmol/h]\n:",sol_PBR_1.y[2][0]*60)
# print("\nFlujo molar a la entrada de Heptadecano [kmol/h]\n:",sol_PBR_1.y[3][0]*60)
# print("\nFlujo molar a la entrada de Octadecano [kmol/h]\n:",sol_PBR_1.y[4][0]*60)
# print("\nFlujo molar a la entrada de Agua [kmol/h]\n:",sol_PBR_1.y[5][0]*60)
# print("\nFlujo molar a la entrada de CO [kmol/h]\n:",sol_PBR_1.y[6][0]*60)


# #FLUJOS MOLARES DE CADA ESPECIE A LA SALIDA
# print("\nFlujo molar a la salida de Acido Estearico [kmol/h]\n:",sol_PBR_1.y[0][-1]*60) #kmol/h
# print("\nFlujo molar a la salida de Hidrogeno [kmol/h]\n:",sol_PBR_1.y[1][-1]*60)
# print("\nFlujo molar a la salida de Estearil Alcohol [kmol/h]\n:",sol_PBR_1.y[2][-1]*60)
# print("\nFlujo molar a la salida de Heptadecano [kmol/h]\n:",sol_PBR_1.y[3][-1]*60)
# print("\nFlujo molar a la salida de Octadecano [kmol/h]\n:",sol_PBR_1.y[4][-1]*60)
# print("\nFlujo molar a la salida de Agua [kmol/h]\n:",sol_PBR_1.y[5][-1]*60)
# print("\nFlujo molar a la salida de CO [kmol/h]\n:",sol_PBR_1.y[6][-1]*60)

# #Flujo Masicos
# print("\nFlujo masico a la entrada de Acido Estearico\n:",sol_PBR_1.y[0][0]*(Reactor.PM_A*1000*60)) # [kg/h]
# print("\nFlujo masico a la entrada de Hidrogeno\n:",sol_PBR_1.y[1][0]*(Reactor.PM_H*1000*60)) # [kg/h]
# print("\nFlujo masico a la entrada de Estearil Alcohol\n:",sol_PBR_1.y[2][0]*(Reactor.PM_B*1000*60)) # [kg/h]
# print("\nFlujo masico a la entrada de Heptadecano\n:",sol_PBR_1.y[3][0]*(Reactor.PM_C*1000*60)) # [kg/h]
# print("\nFlujo masico a la entrada de Octadecano\n:",sol_PBR_1.y[4][0]*(Reactor.PM_D*1000*60)) # [kg/h]
# print("\nFlujo masico a la entrada de Agua\n:",sol_PBR_1.y[5][0]*(Reactor.PM_E*1000*60)) # [kg/h]
# print("\nFlujo masico a la entrada de CO\n:",sol_PBR_1.y[6][0]*(Reactor.PM_CO*1000*60)) # [kg/h]
# total=sol_PBR_1.y[0][0]*(Reactor.PM_A*1000*60)+sol_PBR_1.y[1][0]*(Reactor.PM_H*1000*60)+sol_PBR_1.y[2][0]*(Reactor.PM_B*1000*60)+sol_PBR_1.y[3][0]*(Reactor.PM_C*1000*60)+sol_PBR_1.y[4][0]*(Reactor.PM_D*1000*60)+sol_PBR_1.y[5][0]*(Reactor.PM_E*1000*60)+sol_PBR_1.y[6][0]*(Reactor.PM_CO*1000*60)
# print(total)

# #Flujo Masicos
# print("\nFlujo masico a la salida de Acido Estearico\n:",sol_PBR_1.y[0][-1]*(Reactor.PM_A*1000*60),"[Kg/h]") # [kg/h]
# print("\nFlujo masico a la salida de Hidrogeno\n:",sol_PBR_1.y[1][-1]*(Reactor.PM_H*1000*60),"[Kg/h]") # [kg/h]
# print("\nFlujo masico a la salida de Estearil Alcohol\n:",sol_PBR_1.y[2][-1]*(Reactor.PM_B*1000*60),"[Kg/h]") # [kg/h]
# print("\nFlujo masico a la salida de Heptadecano\n:",sol_PBR_1.y[3][-1]*(Reactor.PM_C*1000*60),"[Kg/h]") # [kg/h]
# print("\nFlujo masico a la salida de Octadecano\n:",sol_PBR_1.y[4][-1]*(Reactor.PM_D*1000*60),"[Kg/h]") # [kg/h]
# print("\nFlujo masico a la salida de Agua\n:",sol_PBR_1.y[5][-1]*(Reactor.PM_E*1000*60),"[Kg/h]") # [kg/h]
# print("\nFlujo masico a la salida de CO\n:",sol_PBR_1.y[6][-1]*(Reactor.PM_CO*1000*60),"[Kg/h]") # [kg/h]
# total=sol_PBR_1.y[0][-1]*(Reactor.PM_A*1000*60)+sol_PBR_1.y[1][-1]*(Reactor.PM_H*1000*60)+sol_PBR_1.y[2][-1]*(Reactor.PM_B*1000*60)+sol_PBR_1.y[3][-1]*(Reactor.PM_C*1000*60)+sol_PBR_1.y[4][-1]*(Reactor.PM_D*1000*60)+sol_PBR_1.y[5][-1]*(Reactor.PM_E*1000*60)+sol_PBR_1.y[6][-1]*(Reactor.PM_CO*1000*60)
# print(total)

# # SALIDA DE PRESION
# print("\nPresion a la salida de Reactor [Bar]\n:",sol_PBR_1.y[7][-1])

# # TEMPERATURA DE SALIDA
# print("\nTemperatura a la salida de Reactor [K]\n:",sol_PBR_1.y[8][-1])
# print("\nTemperatura a la salida de Reactor [K]\n:",sol_PBR_1.y[9][-1])

# # DIMENSIONAMIENTO DE REACTOR
# V_reactor=Reactor.W_tot_su/Reactor.rc
# L_reactor=V_reactor/Reactor.area
# print("Volumen de Reacor",V_reactor,"m^3,\n","Longitud de Reactor",L_reactor,"m,\n","Diametro de Reactor",Reactor.di,"m")

stop = timeit.default_timer()
print('Time: ', stop - start)