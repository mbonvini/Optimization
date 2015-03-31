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
    path_csv = os.path.join(curr_dir,"..","..","Data","DataWeather.csv")
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
    
    return (Tamb_new, Tgnd_new, sRadS_new, sRadN_new, sRadW_new, sRadE_new, ihg_new)

def run_simulation():
    """
    This function runs a simulation that uses inputs data series
    """
    
    # Get the weather data for the building model
    time = np.linspace(0, 3600*6*24, 24*7*6, True)
    (Tamb, Tgnd, sRadS, sRadN, sRadW, sRadE, ihg) = getData(time, plot = False)
    
    # get current directory
    curr_dir = os.path.dirname(os.path.abspath(__file__));
    
    # compile FMU
    path = os.path.join(curr_dir,"..","..","Models","ElectricalNetwork.mop")
    jmu_model = compile_jmu('ElectricNetwork.RCBuildingModel8', path)

    # Load the model instance into Python
    model = JMUModel(jmu_model)
    
    # Solve the DAE initialization system
    model.initialize()
    
    # Build input trajectory matrix for use in simulation
    #  u[1] v_IG_Offices [W/m2]
    #  u[2] v_Tamb
    #  u[3] v_Tgnd
    #  u[4] v_solGlobFac_E [W/m2]
    #  u[5] v_solGlobFac_N [W/m2]
    #  u[6] v_solGlobFac_S [W/m2]
    #  u[7] v_solGlobFac_W [W/m2]
    u = np.transpose(np.vstack((time, ihg, Tamb, Tgnd, sRadE, sRadN, sRadS, sRadW)))
    
    # Solve the DAE initialization system
    model.initialize()
    
    # Simulate
    res = model.simulate(input=(['u[1]', 'u[2]', 'u[3]', 'u[4]', 'u[5]', 'u[6]', 'u[7]'], u), start_time = time[0], final_time = time[-1])
    
    return res

def plot_sim_res(res):
    """
    This function plots the results of a simulation
    """
    time = res["time"]
    Tbui = res["Tmix"]
    Tamb = res["u[2]"]
    
    fig = plt.figure()
    ax = fig.add_subplot(111) 
    ax.plot(time/3600, Tbui, 'r', alpha=0.5, linewidth=2)
    ax.plot(time/3600, np.ones(len(time))*(273.15 + 20), 'b', alpha=0.5, linewidth=2)
    ax.plot(time/3600, np.ones(len(time))*(273.15 + 24), 'b', alpha=0.5, linewidth=2)
    ax.plot(time/3600, Tamb, 'g', alpha=0.5, linewidth=2)
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('Temperature [K]')
    ax.set_title("Temperature of the return air")
    
    plt.show()
    
    return

if __name__ == '__main__':
    
    sim_res = run_simulation()
    plot_sim_res(sim_res)