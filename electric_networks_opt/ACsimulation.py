'''
Created on Apr 2, 2014

@author: marco
'''
import os

from pymodelica import compile_jmu
from pymodelica import compile_fmux
from pyjmi import JMUModel
from pyjmi import CasadiModel

import matplotlib.pyplot as plt
import numpy as np

def getData(new_time, plot = False):
    """
    This function get data from CSV files
    """
    # get current directory
    curr_dir = os.path.dirname(os.path.abspath(__file__));
    path_csv = os.path.join(curr_dir,"..","Data","DataExample.csv")
    data = np.loadtxt(open(path_csv,"rb"), delimiter=",", skiprows=1)
    
    # Get time vector
    time = data[:, 0]
    
    # Get data and normalize them
    pv   = data[:, 1:4]
    pv   = pv/np.max(pv)
    bldg = -data[:, 4:]
    bldg = bldg/np.max(bldg)
    bldg = bldg + 0.015*np.ones(np.shape(bldg))
    bldg = bldg/np.max(bldg)
    
    # interpolate
    pv_new = np.zeros((len(new_time),3))
    bldg_new = np.zeros((len(new_time),3))
    for i in range(3):
        pv_new[:,i]   = np.interp(new_time, time, pv[:,i])
        bldg_new[:,i] = np.interp(new_time, time, bldg[:,i])
    
    # plot if requested 
    if plot:
        figPv = plt.figure()
        ax = figPv.add_subplot(211) 
        ax.plot(new_time/3600, pv_new, 'r', alpha=0.5, linewidth=2)
        ax.set_xlabel('Time [hours]')
        ax.set_ylabel('$P/P_{N} \ [\cdot]$ ')
        ax.set_title("Normalized power generated by PV panels")
        ax.set_xlim([0, 24])
        ax.set_ylim([0, 1])
        
        ax = figPv.add_subplot(212) 
        ax.plot(new_time/3600, bldg_new, 'b', alpha=0.5, linewidth=2)
        ax.set_xlabel('Time [hours]')
        ax.set_ylabel('$P/P_{N} \ [\cdot] $ ')
        ax.set_title("Normalized power consumed by buildings")
        ax.set_xlim([0, 24])
        ax.set_ylim([0, 1])
        
        figAll = plt.figure()
        ax = figAll.add_subplot(111)
        ax.plot(new_time/3600, pv_new[:,0], 'r', alpha=0.5, linewidth=2, label="PVs")
        ax.plot(new_time/3600, pv_new[:,1:], 'r', alpha=0.5, linewidth=2)
        ax.plot(new_time/3600, bldg_new[:,0], 'b', alpha=0.5, linewidth=2, label="Buildings")
        ax.plot(new_time/3600, bldg_new[:,1:], 'b', alpha=0.5, linewidth=2)
        ax.set_xlabel('Time [hours]')
        ax.set_ylabel('P/Pmax [$\cdot$]')
        ax.set_title("Normalized power production and consumption")
        ax.set_xlim([0, 24])
        ax.set_ylim([0, 1])
        legend = ax.legend(loc="upper left")
        legend.draggable(state=True)
        
        plt.show()
    
    return (pv_new, bldg_new)

def run_simulation_with_inputs(time, price, pv, bldg, plot = False, usePV = True):
    """
    This function runs a simulation that uses inputs data series
    """
    
    # get current directory
    curr_dir = os.path.dirname(os.path.abspath(__file__));
    
    # compile FMU
    path = os.path.join(curr_dir,"..","Models","ElectricalNetwork.mop")
    jmu_model = compile_jmu('ElectricNetwork.ACnetwork', path)

    # Load the model instance into Python
    model = JMUModel(jmu_model)
    
    # create input data series for price and current battery
    Npoints = len(time)
    
    # for the simulation no power flow in the battery
    P  = np.zeros(Npoints)
    Q  = np.zeros(Npoints)
    
    # if pv panels are not used then remove power
    if usePV == False:
        pv = np.zeros(np.shape(pv))
    
    # Build input trajectory matrix for use in simulation
    u = np.transpose(np.vstack((t_data, P, Q, price, -np.squeeze(bldg[:,0]), -np.squeeze(bldg[:,1]), -np.squeeze(bldg[:,2]), \
                                np.squeeze(pv[:,0]), np.squeeze(pv[:,1]), np.squeeze(pv[:,2]))))
    
    # Solve the DAE initialization system
    model.initialize()
    
    # Simulate
    res = model.simulate(input=(['P_batt', 'Q_batt', 'price', 'P_bldg1', 'P_bldg2', 'P_bldg3', 'P_pv1', 'P_pv2', 'P_pv3'], u), start_time=0., final_time=24.0*3600.0)
    
    # Extract variable profiles
    if plot:
        plotFunction(res)
        
    return res
    
def plotFunction(res, res_opt=None):
    
    time = res["time"]/3600.0
    price= res["price"]
    Vs   = res["S.Vrms"]
    V1   = res["bldg1.Vrms"]
    V2   = res["bldg2.Vrms"]
    V3   = res["bldg3.Vrms"]
    P_batt = res["batt.P"]
    Q_batt = res["batt.Q"]
    SOC    = res["batt.SOC"]
    E      = res["E"]
    Money  = res["Money"]
    Balance= res["Balance"]
    Ebal   = res["Ebal"]
    
    if res_opt != None:
        time_o = res_opt["time"]/3600.0
        V1_o   = res_opt["bldg1.Vrms"]
        V2_o   = res_opt["bldg2.Vrms"]
        V3_o  = res_opt["bldg3.Vrms"]
        P_batt_o = res_opt["batt.P"]
        Q_batt_o = res_opt["batt.Q"]
        SOC_o    = res_opt["batt.SOC"]
        E_o      = res_opt["E"]
        Money_o  = res_opt["Money"]
        Balance_o= res_opt["Balance"]
        Ebal_o   = res_opt["Ebal"]
        #Action_o = res_opt["D.y"]
        #IntAct_o = res_opt["action"]
    
    figPrice = plt.figure()
    ax = figPrice.add_subplot(111) 
    ax.plot(time, price, 'r', label='$Price$')
    ax.set_xlim([0, 24])
    ax.set_ylim([0.1, 0.25])
    ax.set_xlabel('time [hoyrs]')
    ax.set_ylabel('Energy price [$/kWh]')
    ax.legend()
    
    figV = plt.figure()
    ax = figV.add_subplot(111) 
    if res_opt == None:
        ax.plot(time, Vs, 'k', label='$Vs$')
        ax.fill_between(time, Vs*0.95, Vs*1.05, facecolor='grey', alpha=0.2)
        ax.plot(time, V1, 'r', label='$V_1$')
        ax.plot(time, V2, 'g', label='$V_2$')
        ax.plot(time, V3, 'b', label='$V_3$')
    else:
        ax.plot(time, Vs, 'k', label='$Vs$')
        ax.fill_between(time, Vs*0.95, Vs*1.05, facecolor='grey', alpha=0.2)
        ax.plot(time, V1, 'r--', label='$V_1$')
        ax.plot(time, V2, 'g--', label='$V_2$')
        ax.plot(time, V3, 'b--', label='$V_3$')
        ax.plot(time_o, V1_o, 'r', label='$V_1^{*}$')
        ax.plot(time_o, V2_o, 'g', label='$V_2^{*}$')
        ax.plot(time_o, V3_o, 'b', label='$V_3^{*}$')
    ax.set_xlim([0, 24])
    ax.set_xlabel('time [hours]')
    ax.set_ylabel('Voltage [V]')
    ax.legend(ncol=4, loc="upper center")
    
    figBatt = plt.figure()
    ax = figBatt.add_subplot(211)
    if res_opt == None:
        ax.plot(time, P_batt, 'r', label='$P_{BATT}$')
        ax.plot(time, Q_batt, 'g', label='$Q_{BATT}$')
    else:
        ax.plot(time, P_batt, 'r--', label='$P_{BATT}$')
        ax.plot(time, Q_batt, 'g--', label='$Q_{BATT}$')
        ax.plot(time_o, P_batt_o, 'r', label='$P_{BATT}^{*}$')
        ax.plot(time_o, Q_batt_o, 'g', label='$Q_{BATT}^{*}$')
    ax.set_xlim([0, 24])
    ax.set_xlabel('time [hours]')
    ax.set_ylabel('Power fed into the battery [W]')
    ax.legend(ncol=2, loc="upper center")
    
    ax = figBatt.add_subplot(212)
    if res_opt == None:
        ax.plot(time, SOC, 'k', label='$SOC$')
    else:
        ax.plot(time, SOC, 'k--', label='$SOC$')
        ax.plot(time_o, SOC_o, 'k', label='$SOC^{*}$')
    ax.set_xlim([0, 24])
    ax.set_xlabel('time [hours]')
    ax.set_ylabel('State of Charge [$\cdot$]')
    ax.legend(ncol=1, loc="upper right")
    
    figEM = plt.figure()
    ax = figEM.add_subplot(211) 
    if res_opt == None:
        ax.plot(time, E, 'b', label='$E_{BATT}$')
    else:
        ax.plot(time, E, 'b--', label='$E_{BATT}$')
        ax.plot(time_o, E_o, 'b', label='$E_{BATT}^{*}$')
    ax.set_xlim([0, 24])
    ax.set_xlabel('time [hours]')
    ax.set_ylabel('Energy [kWh]')
    ax.legend(loc="upper left")
    
    ax = figEM.add_subplot(212)
    if res_opt == None:
        ax.plot(time, Money, 'b', label='$Money$')
    else:
        ax.plot(time, Money, 'b--', label='$Money$')
        ax.plot(time_o, Money_o, 'b', label='$Money^{*}$')
    ax.set_xlim([0, 24])
    ax.set_xlabel('time [hours]')
    ax.set_ylabel('Money [$]')
    ax.legend(loc="upper left")
    
    figACTION = plt.figure()
    ax = figACTION.add_subplot(211)
    if res_opt == None:
        ax.plot(time, Balance, 'b', label='$\sum P$')
    else:
        ax.plot(time, Balance, 'b', label='$\sum P$')
    ax.set_xlim([0, 24])
    ax.set_xlabel('time [hours]')
    ax.set_ylabel('Sum of the Powers [W]')
    ax.legend(loc="upper left")
    
    ax = figACTION.add_subplot(212)
    if res_opt == None:
        ax.plot(time, Ebal, 'b', label='$\int \sum P$')
    else:
        ax.plot(time_o, Ebal_o, 'b', label='$\int \sum P$')
    ax.set_xlim([0, 24])
    ax.set_xlabel('time [hours]')
    ax.set_ylabel('Integral of power balance [J]')
    ax.legend(loc="upper left")
    
def plotComparison(res_noPV, res_PV, res_sim_PV):
    time = res_noPV["time"]/3600.0
    price= res_noPV["price"]
    Vs   = res_noPV["S.Vrms"]
    V1   = res_noPV["bldg1.Vrms"]
    V2   = res_noPV["bldg2.Vrms"]
    V3   = res_noPV["bldg3.Vrms"]
    SOC  = res_noPV["batt.SOC"]
    
    time_pv = res_PV["time"]/3600.0
    V1_pv   = res_PV["bldg1.Vrms"]
    V2_pv   = res_PV["bldg2.Vrms"]
    V3_pv   = res_PV["bldg3.Vrms"]
    SOC_pv  = res_PV["batt.SOC"]
    
    time_s_pv = res_sim_PV["time"]/3600.0
    V1_s_pv   = res_sim_PV["bldg1.Vrms"]
    V2_s_pv   = res_sim_PV["bldg2.Vrms"]
    V3_s_pv   = res_sim_PV["bldg3.Vrms"]
    
    figCmp = plt.figure()
    ax = figCmp.add_subplot(212)
    ax.plot(time, SOC, 'r', label='$SOC_{No PV}$')
    ax.plot(time_pv, SOC_pv, 'b', label='$SOC_{PV}$')
    ax.fill_between([6,12], [0,0], [1,1], facecolor='#CC3300', alpha=0.2, linewidth=0)
    ax.fill_between([12,13], [0,0], [1,1], facecolor='#CC3300', alpha=0.3, linewidth=0)
    ax.fill_between([13,16], [0,0], [1,1], facecolor='#CC3300', alpha=0.6, linewidth=0)
    ax.fill_between([16,18], [0,0], [1,1], facecolor='#CC3300', alpha=0.5, linewidth=0)
    ax.fill_between([18,19], [0,0], [1,1], facecolor='#CC3300', alpha=0.4, linewidth=0)
    ax.fill_between([19,22], [0,0], [1,1], facecolor='#CC3300', alpha=0.2, linewidth=0)
    
    ax.set_xlim([0, 24])
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('State of charge [1]')
    legend = ax.legend(loc="upper left")
    legend.draggable(state=True)
    
    ax = figCmp.add_subplot(211)
    ax.plot(time, Vs, 'k')
    ax.fill_between(time, Vs*0.95, Vs*1.05, facecolor='grey', alpha=0.2)
    ax.plot(time, V1, 'r', label='$V_{No PV}$', alpha=0.5)
    ax.plot(time, V2, 'r', alpha=0.5)
    ax.plot(time, V3, 'r', alpha=0.5)
    ax.plot(time_pv, V1_pv, 'b', label='$V_{PV}^{MPC}$')
    ax.plot(time_pv, V2_pv, 'b')
    ax.plot(time_pv, V3_pv, 'b')
    ax.plot(time_s_pv, V1_s_pv, 'b--', label='$V_{PV}$', alpha=0.5)
    ax.plot(time_s_pv, V2_s_pv, 'b--', alpha=0.5)
    ax.plot(time_s_pv, V3_s_pv, 'b--', alpha=0.5)
    ax.set_xlim([0, 24])
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('RMS Voltage [V]')
    legend = ax.legend(loc="upper left")
    legend.draggable(state=True)

def run_optimization(sim_res, time, price, pv, bldg, plot = True, usePV = True):
    """
    This function runs an optimization problem
    """
    
    from pyjmi.optimization.casadi_collocation import MeasurementData
    from collections import OrderedDict

    # get current directory
    curr_dir = os.path.dirname(os.path.abspath(__file__));
    
    # if pv panels are not used then remove power
    if usePV == False:
        pv = np.zeros(np.shape(pv))
    
    # compile FMU
    path = os.path.join(curr_dir,"..","Models","ElectricalNetwork.mop")
    model_name = compile_fmux('ElectricNetwork.ACnetworkBatteryMngmtOpt_Money', path, compiler_options={"enable_variable_scaling":True})
    
    # Load the model
    model_casadi = CasadiModel(model_name) 
    
    # Get the inputs that should be eliminated from the optimization variables
    eliminated = OrderedDict()
    data_price = np.vstack([t_data, price])
    eliminated['price'] = data_price
    
    #data_Q  = np.vstack([t_data, np.zeros(Npoints)])
    #eliminated['Q_batt'] = data_Q
    
    data_pv1 = np.vstack([t_data, np.squeeze(pv[:,0])])
    eliminated['P_pv1'] = data_pv1
    data_pv2 = np.vstack([t_data, np.squeeze(pv[:,1])])
    eliminated['P_pv2'] = data_pv2
    data_pv3 = np.vstack([t_data, np.squeeze(pv[:,2])])
    eliminated['P_pv3'] = data_pv3
    
    data_bldg1 = np.vstack([t_data, -np.squeeze(bldg[:,0])])
    eliminated['P_bldg1'] = data_bldg1
    data_bldg2 = np.vstack([t_data, -np.squeeze(bldg[:,1])])
    eliminated['P_bldg2'] = data_bldg2
    data_bldg3 = np.vstack([t_data, -np.squeeze(bldg[:,2])])
    eliminated['P_bldg3'] = data_bldg3
    
    measurement_data = MeasurementData(eliminated = eliminated)
    
    # define the optimization problem
    opts = model_casadi.optimize_options()
    opts['n_e'] = 60
    opts['measurement_data'] = measurement_data
    opts['init_traj'] = sim_res.result_data
    
    # get the results of the optimization
    res = model_casadi.optimize(options = opts)
    
    if plot:
        plotFunction(sim_res, res)
    
    return res

if __name__ == '__main__':
    #run_simulation()
    #res = run_simulation_with_inputs()
    #run_optimization(res)
    
    # Define the points over which the inputs will be provided
    # One point every 10 minutes
    Npoints = 24*6+1
    
    # Time vector
    t_data = np.linspace(0.0, 24*3600, Npoints)
    
    # Define the price for the electricity
    t_data = np.linspace(0.0, 24*3600, Npoints)
    t_price = np.array([0.0, 6.0, 8.0, 12.0, 13.0, 16.0, 18.0, 19.0, 22.0, 24.0])*3600.0
    price =   np.array([0.5, 0.5, 0.6, 0.8,  1.0,  1.1,  0.8,  0.7,  0.5,  0.5])
    price = 0.22*np.interp(t_data, t_price, price)
    
    # Time series for buildings and PVs
    (pv, bldg) = getData(t_data, plot = True)

    # Run the simulation NO PVs
    res_sim_noPV = run_simulation_with_inputs(t_data, price, pv, bldg, plot = False, usePV = False)
    
    # Run the optimization NO PVs
    res_opt_noPV = run_optimization(res_sim_noPV, t_data, price, pv, bldg, plot = False, usePV = False)
    
    # Run the simulation NO PVs
    res_sim_PV = run_simulation_with_inputs(t_data, price, pv, bldg, plot = False, usePV = True)
    
    # Run the optimization NO PVs
    res_opt_PV = run_optimization(res_sim_PV, t_data, price, pv, bldg, plot = False, usePV = True)
    
    # Show and compare results
    plotFunction(res_sim_noPV, res_opt_noPV)
    plotFunction(res_sim_PV, res_opt_PV)
    
    # plot and compare results
    plotComparison(res_opt_noPV, res_opt_PV, res_sim_PV)
    
    plt.show()