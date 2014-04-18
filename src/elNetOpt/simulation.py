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
    path_csv = os.path.join(curr_dir,"..","..","Data","DataExample.csv")
    data = np.loadtxt(open(path_csv,"rb"), delimiter=",", skiprows=1)
    
    # Get time vector
    time = data[:, 0]
    
    # Get data and normalize them
    pv   = data[:, 1:4]
    pv   = pv/np.max(pv)
    bldg = -data[:, 4:]
    bldg = bldg/np.max(bldg)
    bldg = bldg + 0.5*np.ones(np.shape(bldg))
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
        ax.plot(new_time, pv_new, 'r', alpha=0.3)
        ax.set_xlabel('Time [hours]')
        ax.set_ylabel('Power [W]')
        ax.set_xlim([0, 86400])
        ax.set_ylim([0, 1])
        
        ax = figPv.add_subplot(212) 
        ax.plot(new_time, bldg_new, 'b', alpha=0.3)
        ax.set_xlabel('Time [hours]')
        ax.set_ylabel('Power [W]')
        ax.set_xlim([0, 86400])
        ax.set_ylim([0, 1])
        
        plt.show()
    
    return (pv_new, bldg_new)
    
def run_simulation(plot = False):
    """
    This function runs a simple simulation without input data
    """
    
    # get current directory
    curr_dir = os.path.dirname(os.path.abspath(__file__));
    
    # compile FMU
    path = os.path.join(curr_dir,"..","..","Models","ElectricalNetwork.mop")
    jmu_model = compile_jmu('ElectricNetwork.NetworkSim', path)

    # Load the model instance into Python
    model = JMUModel(jmu_model)
  
    # Solve the DAE initialization system
    model.initialize()
    
    # Simulate
    res = model.simulate(start_time=0., final_time=24.0*3600.0)
    
    # Extract variable profiles
    Vs_init_sim = res['n.Vs']
    V1_init_sim = res['n.V1']
    V2_init_sim = res['n.V2']
    V3_init_sim = res['n.V3']
    E_init_sim = res['n.E']
    SOC_init_sim = res['n.SOC']
    Money_init_sim = res['n.Money']
    price_init_sim = res['n.price']
    t_init_sim = res['time']
    
    # plot results
    if plot:
        plotFunction(t_init_sim, Vs_init_sim, V1_init_sim, V2_init_sim, \
                 V3_init_sim, E_init_sim, SOC_init_sim, Money_init_sim, price_init_sim)

def run_simulation_with_inputs(time, price, pv, bldg, plot = False):
    """
    This function runs a simulation that uses inputs data series
    """
    
    # get current directory
    curr_dir = os.path.dirname(os.path.abspath(__file__));
    
    # compile FMU
    path = os.path.join(curr_dir,"..","..","Models","ElectricalNetwork.mop")
    jmu_model = compile_jmu('ElectricNetwork.Network', path)

    # Load the model instance into Python
    model = JMUModel(jmu_model)
    
    # create input data series for price and current battery
    Npoints = len(time)
    
    # for the simulation no current flow
    Ibatt  = np.zeros(Npoints)
    
    # Build input trajectory matrix for use in simulation
    u = np.transpose(np.vstack((t_data, Ibatt, price, np.squeeze(pv[:,0]), np.squeeze(pv[:,1]), \
                                np.squeeze(pv[:,2]), np.squeeze(bldg[:,0]), np.squeeze(bldg[:,1]), np.squeeze(bldg[:,2]))))
    
    # Solve the DAE initialization system
    model.initialize()
    
    # Simulate
    res = model.simulate(input=(['Ibatt', 'price', 'pv1', 'pv2', 'pv3', 'bldg1', 'bldg2', 'bldg3'], u), start_time=0., final_time=24.0*3600.0)
    
    # Extract variable profiles
    Vs_init_sim = res['Vs']
    V1_init_sim = res['V1']
    V2_init_sim = res['V2']
    V3_init_sim = res['V3']
    E_init_sim = res['E']
    SOC_init_sim = res['SOC']
    Money_init_sim = res['Money']
    price_init_sim = res['price']
    t_init_sim = res['time']
    
    # plot results
    if plot:
        plotFunction(t_init_sim, Vs_init_sim, V1_init_sim, V2_init_sim, \
                 V3_init_sim, E_init_sim, SOC_init_sim, Money_init_sim, price_init_sim)
    
    return res

def run_optimization(sim_res, time, price, pv, bldg):
    """
    This function runs an optimization problem
    """
    
    from pyjmi.optimization.casadi_collocation import MeasurementData
    from collections import OrderedDict

    # get current directory
    curr_dir = os.path.dirname(os.path.abspath(__file__));
    
    # compile FMU
    path = os.path.join(curr_dir,"..","..","Models","ElectricalNetwork.mop")
    model_name = compile_fmux('ElectricNetwork.NetworkBatteryMngmtOpt_Money', path, compiler_options={"enable_variable_scaling":True})
    
    # Load the model
    model_casadi = CasadiModel(model_name) 
    
    # Get the inputs that should be eliminated from the optimization variables
    eliminated = OrderedDict()
    data_price = np.vstack([t_data, price])
    eliminated['price'] = data_price
    
    data_pv1 = np.vstack([t_data, np.squeeze(pv[:,0])])
    eliminated['pv1'] = data_pv1
    data_pv2 = np.vstack([t_data, np.squeeze(pv[:,1])])
    eliminated['pv2'] = data_pv2
    data_pv3 = np.vstack([t_data, np.squeeze(pv[:,2])])
    eliminated['pv3'] = data_pv3
    
    data_bldg1 = np.vstack([t_data, np.squeeze(bldg[:,0])])
    eliminated['bldg1'] = data_bldg1
    data_bldg2 = np.vstack([t_data, np.squeeze(bldg[:,1])])
    eliminated['bldg2'] = data_bldg2
    data_bldg3 = np.vstack([t_data, np.squeeze(bldg[:,2])])
    eliminated['bldg3'] = data_bldg3
    
    measurement_data = MeasurementData(eliminated = eliminated)
    
    # define the optimization problem
    opts = model_casadi.optimize_options()
    opts['n_e'] = 50
    opts['measurement_data'] = measurement_data
    opts['init_traj'] = sim_res.result_data
    
    # get the results of the optimization
    res = model_casadi.optimize(options = opts)
    
    plotCompare(sim_res, res)
    
def plotCompare(simulation, optimization):
    # Extract variable profiles
    Vs_sim = simulation['Vs']
    V1_sim = simulation['V1']
    V2_sim = simulation['V2']
    V3_sim = simulation['V3']
    E_sim = simulation['E']
    SOC_sim = simulation['SOC']
    Money_sim = simulation['Money']
    price_sim = simulation['price']
    t_sim = simulation['time']/3600.0
    
    # Extract variable profiles
    Vs_opt = optimization['Vs']
    V1_opt = optimization['V1']
    V2_opt = optimization['V2']
    V3_opt = optimization['V3']
    E_opt = optimization['E']
    SOC_opt = optimization['SOC']
    Money_opt = optimization['Money']
    price_opt = optimization['price']
    t_opt = optimization['time']/3600.0
    
    plt.figure(1)
    plt.clf()
    plt.hold(True)
    plt.subplot(311)
    plt.plot(t_sim, SOC_sim)
    plt.plot(t_opt, SOC_opt)
    plt.grid()
    plt.ylabel('SOC')
    
    plt.subplot(312)
    plt.plot(t_sim, E_sim)
    plt.plot(t_opt, E_opt)
    plt.grid()
    plt.ylabel('Energy')
    
    plt.subplot(313)
    plt.plot(t_sim, Money_sim)
    plt.plot(t_opt, Money_opt)
    plt.grid()
    plt.ylabel('Money')
    
    figV = plt.figure()
    ax = figV.add_subplot(111) 
    ax.plot(t_sim, Vs_sim, 'k', alpha=1)
    ax.plot(t_sim, V1_sim, 'r', alpha=0.7)
    ax.plot(t_sim, V2_sim, 'r', alpha=0.7)
    ax.plot(t_sim, V3_sim, 'r', alpha=0.7)

    ax.plot(t_opt, V1_opt, 'b', alpha=0.7)
    ax.plot(t_opt, V2_opt, 'b', alpha=0.7)
    ax.plot(t_opt, V3_opt, 'b', alpha=0.7)
    ax.fill_between(t_opt, 4800*0.99*np.ones(np.shape(t_opt)), 4800*1.01*np.ones(np.shape(t_opt)), facecolor='green', alpha=0.3)
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('Voltage [W]')
    
    plt.show()

def plotFunction(time, Vs, V1, V2, V3, E, SOC, Money, price):
    
    time = time/3600.0
    
    plt.figure(1)
    plt.clf()
    plt.hold(True)
    plt.subplot(411)
    plt.plot(time, Vs)
    plt.plot(time, V1)
    plt.plot(time, V2)
    plt.plot(time, V3)
    plt.grid()
    plt.ylabel('Voltages')
    
    plt.subplot(412)
    plt.plot(time, E)
    plt.grid()
    plt.ylabel('Energy')
    
    plt.subplot(413)
    plt.hold(True)
    plt.plot(time, Money)
    plt.grid()
    plt.ylabel('Money')
    
    plt.subplot(414)
    plt.plot(time, SOC)
    plt.grid()
    plt.ylabel('State of Charge')
    
    figPrice = plt.figure()
    ax = figPrice.add_subplot(111) 
    ax.plot(time, price, 'r', label='$Price$')
    ax.set_xlim([0, 24])
    ax.set_ylim([0.1, 0.25])
    ax.set_xlabel('time [hoyrs]')
    ax.set_ylabel('Energy price [$/kWh]')
    ax.legend()
    
    plt.show()

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
    (pv, bldg) = getData(t_data, plot = False)
    
    # Run the simulation
    res = run_simulation_with_inputs(t_data, price, pv, bldg, plot = True)
    
    # Run the optimization
    run_optimization(res, t_data, price, pv, bldg)