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
    ihg = (P_plug_m2+P_light_m2)*np.ones(len(time))
    
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
    ihg_new = np.interp(new_time, time, ihg)
    
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
    
    return (Tamb_new, Tgnd_new, sRadS_new, sRadN_new, sRadW_new, sRadE_new, ihg_new, price)

def run_simulation():
    """
    This function runs a simulation that uses inputs data series
    """
    
    # Get the weather data and other inputs for the building model
    time = np.linspace(0, 3600*24*6, 24*6*6, True)
    (Tamb, Tgnd, sRadS, sRadN, sRadW, sRadE, ihg, price) = getData(time, plot = False)
    P_hvac = -1e5*np.ones(len(time))
    
    # Get current directory
    curr_dir = os.path.dirname(os.path.abspath(__file__));
    
    # Compile FMU
    path = os.path.join(curr_dir,"..","Models","ElectricalNetwork.mop")
    fmu_model = compile_fmu('ElectricNetwork.District', path)

    # Load the model instance into Python
    model = load_fmu(fmu_model)
        
    # Build input trajectory matrix for use in simulation
    #
    # price "Price of kWh"; 
    # P_hvac_1 "Cooling/Heating power supplied by the HVAC system in building 1";
    # P_hvac_2 "Cooling/Heating power supplied by the HVAC system in building 2";
    # P_hvac_3 "Cooling/Heating power supplied by the HVAC system in building 3";
    # ihg_1 "Internal heat gains for building 1";
    # ihg_2 "Internal heat gains for building 2";
    # ihg_3 "Internal heat gains for building 3";
    # Tamb;
    # Tgnd;
    # solGlobFac_E;
    # solGlobFac_N;
    # solGlobFac_S;
    # solGlobFac_W;
    u = np.transpose(np.vstack((time, ihg, ihg, ihg, P_hvac, P_hvac, P_hvac,
                                Tamb, Tgnd, sRadE, sRadN, sRadS, sRadW, price)))
    
    # Simulate
    res = model.simulate(input=(['ihg_1', 'ihg_2', 'ihg_3', 'P_hvac_1', 'P_hvac_2', 'P_hvac_3', 
                                 'Tamb', 'Tgnd', 'solGlobFac_E', 'solGlobFac_N', 'solGlobFac_S', 'solGlobFac_W', 'price'], u), 
                         start_time = time[0], final_time = time[-1])
    
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