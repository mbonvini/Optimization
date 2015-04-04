'''
Created on Apr 1, 2015

@author: marco
'''

import os

from pymodelica import compile_fmu
from pyfmi import load_fmu 
from pyjmi import transfer_optimization_problem 

import matplotlib.pyplot as plt
import numpy as np

def getData(new_time, plot = False):
    """
    This function get data from CSV files
    """
    # get current directory
    curr_dir = os.path.dirname(os.path.abspath(__file__));
    path_csv = os.path.join(curr_dir,"..","Data","DataWeatherSacramento.csv")
    data = np.loadtxt(open(path_csv,"rb"), delimiter=",", skiprows=1)
    
    # Get time vector
    time = data[:, 0]
    time = time - time[0]
    
    # Get data and normalize them
    Tamb = data[:, 1]
    Tgnd = data[:, 2]
    sRadS = data[:, 3]
    sRadN = data[:, 4]
    sRadE = data[:, 5]
    sRadW = data[:, 6]
    ihg = data[:, 7]
    
    # Internal Heat gains are defined here
    P_plug_m2 = 25
    P_light_m2 = 16
    P_occ_m2 = 120/18.6
    ihg = (P_plug_m2+P_light_m2)
    
    # variations to the ihg for each building
    t_ihg = np.array([0.0, 6.0, 8.0, 10, 12.0, 13.0, 16.0, 18.0, 19.0, 22.0, 24.0])*3600.0
    time_ihg = np.hstack((t_ihg, t_ihg + 24*3600, t_ihg + 2*24*3600, t_ihg + 3*24*3600, t_ihg + 4*24*3600,
    t_ihg + 5*24*3600, t_ihg + 6*24*3600))
    ihg_1 = ihg*np.array((0.4, 0.5, 0.6, 1.0, 1.0, 0.8, 1.0, 0.7, 0.6, 0.2, 0.4)*7) + 0.*np.random.rand(len(time_ihg))
    ihg_2 = ihg*np.array((0.4, 0.7, 0.6, 0.5, 1.0, 0.8, 0.6, 0.7, 0.6, 0.2, 0.4)*7) + 0.*np.random.rand(len(time_ihg))
    ihg_3 = ihg*np.array((0.2, 0.5, 1.0, 1.0, 0.9, 0.8, 0.5, 0.7, 0.6, 0.2, 0.2)*7) + 0.*np.random.rand(len(time_ihg))
    ihg_1 = np.interp(new_time, time_ihg, ihg_1)
    ihg_2 = np.interp(new_time, time_ihg, ihg_2)
    ihg_3 = np.interp(new_time, time_ihg, ihg_3)
 
    # Price signal
    t_price = np.array([0.0, 6.0, 8.0, 12.0, 13.0, 16.0, 18.0, 19.0, 22.0, 24.0])*3600.0
    price =   np.array((0.5, 0.5, 0.6, 0.8,  1.0,  1.1,  0.8,  0.7,  0.5,  0.5)*7)
    t_price = np.hstack((t_price, 24*3600 + t_price, 2*24*3600 + t_price, 3*24*3600 + t_price, 
    4*24*3600 + t_price, 5*24*3600 + t_price, 6*24*3600 + t_price))

    price = 0.22*np.interp(new_time, t_price, price)
    
    # Interpolate
    Tamb_new = np.interp(new_time, time, Tamb)
    Tgnd_new = np.interp(new_time, time, Tgnd)
    sRadS_new = np.interp(new_time, time, sRadS)
    sRadN_new = np.interp(new_time, time, sRadN)
    sRadE_new = np.interp(new_time, time, sRadE)
    sRadW_new = np.interp(new_time, time, sRadW)
    
    # Plot if requested 
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(211) 
        ax.plot(new_time/3600, Tamb_new, 'r', alpha=0.5, linewidth=2)
        ax.plot(new_time/3600, Tgnd_new, 'b', alpha=0.5, linewidth=2)
        ax.set_xlabel('Time [hours]')
        ax.set_ylabel('Temperature [K]')
        ax.set_title("External temperature")
        ax.set_xlim([0, 24])
        ax.set_ylim([280, 310])
        
        ax = fig.add_subplot(212) 
        ax.plot(new_time/3600, sRadS_new, 'b', alpha=0.5, linewidth=2)
        ax.plot(new_time/3600, sRadN_new, 'g', alpha=0.5, linewidth=2)
        ax.plot(new_time/3600, sRadW_new, 'r', alpha=0.5, linewidth=2)
        ax.plot(new_time/3600, sRadE_new, 'k', alpha=0.5, linewidth=2)
        ax.set_xlabel('Time [hours]')
        ax.set_ylabel('Solar radiation [W/m2] ')
        ax.set_title("Solar radiation per square meters")
        ax.set_xlim([0, 24*6])
        ax.set_ylim([0, 800])
        
        plt.show()
    
    return (Tamb_new, Tgnd_new, sRadS_new, sRadN_new, sRadW_new, sRadE_new, ihg_1, ihg_2, ihg_3, price)

def plot_sim_res(res):
    """
    This function plots the results of a simulation
    """
    time = res["time"]
    Tbui_1 = res["bui1.Tmix"]
    Tbui_2 = res["bui2.Tmix"]
    Tbui_3 = res["bui3.Tmix"]
    PV_1 = res["pv1.P"]
    PV_2 = res["pv2.P"]
    PV_3 = res["pv3.P"]
    V_1 = res["pv1.Vrms"]
    V_2 = res["pv2.Vrms"]
    V_3 = res["pv3.Vrms"]
    Tamb = res["Tamb"]
    price = res["price"]
    price_max = np.max(price)
    price_min = np.min(price)
 
    fig = plt.figure()
    ax = fig.add_subplot(111) 
    
    t0 = time[0]
    t1 = time[-1]
    T = np.arange(t0, t1, 3600.0)
    price_T = np.interp(T, time, price)
    alpha_max = 0.7
    for i in range(2,len(T)):
        a = (((price_T[i-1]+price_T[i])/2)-price_min)/(price_max - price_min)*alpha_max 
        ax.fill_between([i-1, i], [0,0], [350,350], facecolor="#CC3300", alpha = a, linewidth = 0)  

    ax.plot(time/3600, Tbui_1, 'r', alpha=0.5, linewidth=2)
    ax.plot(time/3600, Tbui_2, 'r', alpha=0.5, linewidth=2)
    ax.plot(time/3600, Tbui_3, 'r', alpha=0.5, linewidth=2)
    ax.plot(time/3600, np.ones(len(time))*(273.15 + 20), 'b', alpha=0.5, linewidth=2)
    ax.plot(time/3600, np.ones(len(time))*(273.15 + 24), 'b', alpha=0.5, linewidth=2)
    ax.plot(time/3600, Tamb, 'g', alpha=0.5, linewidth=2)    
        
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('Temperature [K]')
    ax.set_title("Temperature of the return air")
    ax.set_ylim([285,312])
    ax.set_xlim([time[0],time[-1]/3600])  
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(111) 
    ax.plot(time/3600, PV_1/1e3, 'k', alpha=0.5, linewidth=2)
    ax.plot(time/3600, PV_2/1e3, 'k', alpha=0.5, linewidth=2)
    ax.plot(time/3600, PV_3/1e3, 'k', alpha=0.5, linewidth=2)
    ax.set_xlim([time[0],time[-1]/3600])  
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('Power [kW]')
    ax.set_title("Power generated by PVs")
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.fill_between([time[0]/3600, time[-1]/3600], [4.8*0.95, 4.8*0.95], [4.8*1.05,4.8*1.05], facecolor="#2ecc71", alpha = 0.5, linewidth = 0) 
    ax.plot([time[0]/3600, time[-1]/3600], [4.8, 4.8], color="#2ecc71", linewidth = 2)
    ax.plot(time/3600, V_1/1e3, 'b', alpha=0.5, linewidth=2)
    ax.plot(time/3600, V_2/1e3, 'b', alpha=0.5, linewidth=2)
    ax.plot(time/3600, V_3/1e3, 'b', alpha=0.5, linewidth=2)
    ax.set_xlim([time[0],time[-1]/3600])  
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('Voltage [kV]')
    ax.set_title("Voltage at the PVs and buildings")
    
    return
    
def run_simulation():
    """
    This function runs a simulation that uses inputs data series
    """
    
    # Get the weather data and other inputs for the building model
    time = np.linspace(0, 3600*24, 24*6*6, True)
    (Tamb, Tgnd, sRadS, sRadN, sRadW, sRadE, ihg_1, ihg_2, ihg_3, price) = getData(time, plot = False)
    Tsp = (273.15+22)*np.ones(len(time))
    P_batt = 0*np.ones(len(time))
 
    # Get current directory
    curr_dir = os.path.dirname(os.path.abspath(__file__));
    
    # Compile FMU
    path = os.path.join(curr_dir,"..","Models","ElectricalNetwork.mop")
    fmu_model = compile_fmu('ElectricNetwork.District', path)

    # Load the model instance into Python
    model = load_fmu(fmu_model)
        
    # Build input trajectory matrix for use in simulation
    #
    # Tsp Set point temperature for the buildings in K
    # P_batt power to charge or discharge the battery
    # price "Price of kWh"; 
    # ihg_1 "Internal heat gains for building 1";
    # ihg_2 "Internal heat gains for building 2";
    # ihg_3 "Internal heat gains for building 3";
    # Tamb;
    # Tgnd;
    # solGlobFac_E;
    # solGlobFac_N;
    # solGlobFac_S;
    # solGlobFac_W;
    u = np.transpose(np.vstack((time, ihg_1, ihg_2, ihg_3, Tsp, P_batt, 
                                Tamb, Tgnd, sRadE, sRadN, sRadS, sRadW, price)))
    
    # Simulate
    res = model.simulate(input=(['ihg_1', 'ihg_2', 'ihg_3', 'Tsp', 'P_batt',
                                 'Tamb', 'Tgnd', 'solGlobFac_E', 'solGlobFac_N', 'solGlobFac_S', 'solGlobFac_W', 'price'], u), 
                         start_time = time[0], final_time = time[-1])
    
    return res

def run_optimization(sim_res, opt_problem = 'ElectricNetwork.OptimizationDistrict_E'):
    """
    This function runs an optimization problem
    """
    
    from pyjmi.optimization.casadi_collocation import MeasurementData
    from collections import OrderedDict
    
    # Get the weather data and other inputs for the building model
    time = np.linspace(0, 3600*24, 24*6*6, True)
    (Tamb, Tgnd, sRadS, sRadN, sRadW, sRadE, ihg_1, ihg_2, ihg_3, price) = getData(time, plot = False)
    Tsp = (273.15 + 22)*np.ones(len(time)) 
 
    # get current directory
    curr_dir = os.path.dirname(os.path.abspath(__file__));
    
    # compile FMU
    path = os.path.join(curr_dir,"..","Models","ElectricalNetwork.mop")
    op_model = transfer_optimization_problem(opt_problem, path, compiler_options={"enable_variable_scaling":True})
    
    # Get the inputs that should be eliminated from the optimization variables
    eliminated = OrderedDict()
    
    data_Tsp = np.vstack([time, Tsp])
    eliminated['Tsp'] = data_Tsp
 
    data_ihg_1 = np.vstack([time, ihg_1])
    eliminated['ihg_1'] = data_ihg_1
    
    data_ihg_2 = np.vstack([time, ihg_2])
    eliminated['ihg_2'] = data_ihg_2
    
    data_ihg_3 = np.vstack([time, ihg_3])
    eliminated['ihg_3'] = data_ihg_3
    
    data_Tamb = np.vstack([time, Tamb])
    eliminated['Tamb'] = data_Tamb
    
    data_Tgnd = np.vstack([time, Tgnd])
    eliminated['Tgnd'] = data_Tgnd
    
    data_sRadE = np.vstack([time, sRadE])
    eliminated['solGlobFac_E'] = data_sRadE
    
    data_sRadN = np.vstack([time, sRadN])
    eliminated['solGlobFac_N'] = data_sRadN
    
    data_sRadS = np.vstack([time, sRadS])
    eliminated['solGlobFac_S'] = data_sRadS
    
    data_sRadW = np.vstack([time, sRadW])
    eliminated['solGlobFac_W'] = data_sRadW
   
    data_price = np.vstack([time, price])
    eliminated['price'] = data_price
 
    measurement_data = MeasurementData(eliminated = eliminated)
    
    # define the optimization problem
    opts = op_model.optimize_options()
    opts['n_e'] = 60*2
    opts['measurement_data'] = measurement_data
    opts['init_traj'] = sim_res.result_data
    
    # Get the results of the optimization
    res = op_model.optimize(options = opts)
    
    plot_sim_res(res)
    
    return res

def run():
    """
    This function run a simulation and two different optimization scenario
    """
    # Run simulation and plot first results
    sim_res = run_simulation()
    
    return (sim_res)

if __name__ == '__main__':
    run()
