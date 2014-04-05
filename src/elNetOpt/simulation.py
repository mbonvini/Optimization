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

def run_simulation():
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
    plotFunction(t_init_sim, Vs_init_sim, V1_init_sim, V2_init_sim, \
                 V3_init_sim, E_init_sim, SOC_init_sim, Money_init_sim, price_init_sim)

def run_simulation_with_inputs():
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
    Npoints = 24*6+1 # every 10 minutes
    t_data = np.linspace(0.0, 24*3600, Npoints)
    Ibatt  = np.zeros(Npoints)
    price  = 0.001*(np.ones(Npoints) + 0.3*np.sin(2*np.pi*1/(24*3600)*t_data))
    
    # Build input trajectory matrix for use in simulation
    u = np.transpose(np.vstack((t_data, Ibatt, price)))
    
    # Solve the DAE initialization system
    model.initialize()
    
    # Simulate
    res = model.simulate(input=(['Ibatt','price'], u), start_time=0., final_time=24.0*3600.0)
    
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
    plotFunction(t_init_sim, Vs_init_sim, V1_init_sim, V2_init_sim, \
                 V3_init_sim, E_init_sim, SOC_init_sim, Money_init_sim, price_init_sim)
    
    return res

def run_optimization(sim_res):
    """
    This function runs an optimization problem
    """
    
    from pyjmi.optimization.casadi_collocation import MeasurementData
    from collections import OrderedDict

    # get current directory
    curr_dir = os.path.dirname(os.path.abspath(__file__));
    
    # compile FMU
    path = os.path.join(curr_dir,"..","..","Models","ElectricalNetwork.mop")
    model_name = compile_fmux('ElectricNetwork.NetworkBatteryMngmtOpt_E', path, compiler_options={"enable_variable_scaling":True})
    
    # Load the model
    model_casadi = CasadiModel(model_name)
    
    # Create input data to be eliminated from the optimization inputs
    Npoints = 60 # every 10 minutes
    t_data = np.linspace(0.0, 24*3600, Npoints)
    price  = 0.001*(np.ones(Npoints) + 0.3*np.sin(2*np.pi*1/(24*3600)*t_data))
    
    data_price = np.vstack([t_data, price])
    eliminated = OrderedDict()
    eliminated['price'] = data_price
    
    measurement_data = MeasurementData(eliminated = eliminated)
    
    opts = model_casadi.optimize_options()
    opts['n_e'] = Npoints
    opts['measurement_data'] = measurement_data
    opts['init_traj'] = sim_res.result_data
    
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
    plt.show()

if __name__ == '__main__':
    #run_simulation()
    res = run_simulation_with_inputs()
    run_optimization(res)