'''
Created on Apr 2, 2014

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

def run_simulation(sim_model = 'ElectricNetwork.SingleBuildingElectric', P_hvac = 0.0, P_batt = 0.0, fixpars = None):
    """
    This function runs a simulation that uses inputs data series
    """
    
    # Get the weather data for the building model
    time = np.linspace(0, 3600*24*6, 24*6*6, True)
    (Tamb, Tgnd, sRadS, sRadN, sRadW, sRadE, ihg, price) = getData(time, plot = False)
    P_hvac = P_hvac*np.ones(len(time))
    P_batt = P_batt*np.ones(len(time))
    
    # Get current directory
    curr_dir = os.path.dirname(os.path.abspath(__file__));
    
    # Compile FMU
    path = os.path.join(curr_dir,"..","Models","ElectricalNetwork.mop")
    fmu_model = compile_fmu(sim_model, path)

    # Load the model instance into Python
    model = load_fmu(fmu_model)
    
    # Change parameters depending on the case
    if fixpars:
    	# Set the values for the parameters
		model.reset()
		model.set(fixpars.keys(),fixpars.values())
    
    # Build input trajectory matrix for use in simulation
    #  u[1] v_IG_Offices [W/m2]
    #  u[2] v_Tamb
    #  u[3] v_Tgnd
    #  u[4] v_solGlobFac_E [W/m2]
    #  u[5] v_solGlobFac_N [W/m2]
    #  u[6] v_solGlobFac_S [W/m2]
    #  u[7] v_solGlobFac_W [W/m2]
    #  P_hvac
    #  Price
    u = np.transpose(np.vstack((time, ihg, Tamb, Tgnd, sRadE, sRadN, sRadS, sRadW, P_hvac, price, P_batt)))
    
    # Simulate
    res = model.simulate(input=(['u[1]', 'u[2]', 'u[3]', 'u[4]', 'u[5]', 'u[6]', 'u[7]', 'P_hvac', 'price', 'P_batt'], u), start_time = time[0]-3600*24.0, final_time = time[-1])
    
    return res

def run_optimization(sim_res, opt_problem = 'ElectricNetwork.BuildingMngmtOpt_E', use_battery = False, fixpars = None, n_e = 24):
    """
    This function runs an optimization problem
    """
    
    from pyjmi.optimization.casadi_collocation import MeasurementData
    from collections import OrderedDict
    
    # Get the weather data for the building model
    time = np.linspace(0, 3600*24*6, 24*6*6, True)
    (Tamb, Tgnd, sRadS, sRadN, sRadW, sRadE, ihg, price) = getData(time, plot = False)
    P_batt = 0.0*np.ones(len(time))
    
    # get current directory
    curr_dir = os.path.dirname(os.path.abspath(__file__));
    
    # compile FMU
    path = os.path.join(curr_dir,"..","Models","ElectricalNetwork.mop")
    op_model = transfer_optimization_problem(opt_problem, path, compiler_options={"enable_variable_scaling":True})
    
    # Change parameters depending on the case
    if fixpars:
    	# Set the values for the parameters
		op_model.set(fixpars.keys(),fixpars.values())
    
    # Get the inputs that should be eliminated from the optimization variables
    eliminated = OrderedDict()
    
    data_ihg = np.vstack([time, ihg])
    eliminated['u[1]'] = data_ihg
    
    data_Tamb = np.vstack([time, Tamb])
    eliminated['u[2]'] = data_Tamb
    
    data_Tgnd = np.vstack([time, Tgnd])
    eliminated['u[3]'] = data_Tgnd
    
    data_sRadE = np.vstack([time, sRadE])
    eliminated['u[4]'] = data_sRadE
    
    data_sRadN = np.vstack([time, sRadN])
    eliminated['u[5]'] = data_sRadN
    
    data_sRadS = np.vstack([time, sRadS])
    eliminated['u[6]'] = data_sRadS
    
    data_sRadW = np.vstack([time, sRadW])
    eliminated['u[7]'] = data_sRadW
   
    data_price = np.vstack([time, price])
    eliminated['price'] = data_price
    
    if not use_battery:
    	data_Pbatt = np.vstack([time, P_batt])
    	eliminated['P_batt'] = data_Pbatt
 
    measurement_data = MeasurementData(eliminated = eliminated)
    
    # define the optimization problem
    opts = op_model.optimize_options()
    opts['n_e'] = n_e
    opts['measurement_data'] = measurement_data
    opts['init_traj'] = sim_res.result_data
    opts["IPOPT_options"]["max_iter"] = 10000
    
    # Get the results of the optimization
    res = op_model.optimize(options = opts)
    
    plot_sim_res(res)
    
    return res

def plot_sim_res(res):
    """
    This function plots the results of a simulation
    """
    time = res["time"]
    Tbui = res["Tmix"]
    Tamb = res["u[2]"]
    price = res["price"]
    price_max = np.max(price)
    price_min = np.min(price)
 
    fig = plt.figure()
    ax = fig.add_subplot(111) 
    ax.plot(time/3600, Tbui, 'r', alpha=0.5, linewidth=2)
    ax.plot(time/3600, np.ones(len(time))*(273.15 + 20), 'b', alpha=0.5, linewidth=2)
    ax.plot(time/3600, np.ones(len(time))*(273.15 + 24), 'b', alpha=0.5, linewidth=2)
    ax.plot(time/3600, Tamb, 'g', alpha=0.5, linewidth=2)
    
    t0 = time[0]
    t1 = time[-1]
    T = np.arange(t0, t1, 3600.0)
    price_T = np.interp(T, time, price)
    alpha_max = 0.8
    for i in range(2,len(T)):
        a = (((price_T[i-1]+price_T[i])/2)-price_min)/(price_max - price_min)*alpha_max 
        ax.fill_between([i-1, i], [0,0], [350,350], facecolor="#CC3300", alpha = a, linewidth = 0)  

    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('Temperature [K]')
    ax.set_title("Temperature of the return air")
    ax.set_ylim([285,312])    
    plt.show()
    
    return

def print_details(opt_res_money, opt_res_energy):
    """
    This function prints some details about the different optimziation results
    """
    E_mon = opt_res_money["Money"][-1] - opt_res_money["Money"][0]
    E_en = opt_res_energy["Money"][-1] - opt_res_energy["Money"][0]
    print "Money spent while optimizing energy [$]: {0}".format(E_en)
    print "Money spent while optimizing cost [$]: {0}".format(E_mon)
    print "Savings [$, %]: {0} -- {1}%".format(E_en - E_mon, 100*(E_en - E_mon)/E_en)
    return

def run():
    """
    This function run a simulation and two different optimization scenario
    """
    # Run simulation and plot first results
    sim_res = run_simulation()
    plot_sim_res(sim_res)
    
    # Run optimization that minimizes energy
    opt_res_energy = run_optimization(sim_res, "ElectricNetwork.BuildingMngmtOpt_E")
    
    # Run optimization that minimizes cost
    opt_res_money = run_optimization(sim_res, "ElectricNetwork.BuildingMngmtOpt_Money")
    
    # Show some datails
    print_details(opt_res_money, opt_res_energy)
    
    return (sim_res, opt_res_energy, opt_res_money)

if __name__ == '__main__':
    run()
    