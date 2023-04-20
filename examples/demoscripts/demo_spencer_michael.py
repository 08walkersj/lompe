#%%
import numpy as np
import pandas as pd
from datetime import timedelta
from lompe.utils.conductance import hardy_EUV
import apexpy
import lompe
import matplotlib.pyplot as plt
import copy
import os
import scipy.linalg

#%%
plt.ioff()

#%%
conductance_functions = True

event = '2012-04-05'
apex = apexpy.Apex(int(event[0:4]), refh = 110)

supermagfn = '/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/lompe/examples/sample_dataset/20120405_supermag.h5'
superdarnfn = '/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/lompe/examples/sample_dataset/20120405_superdarn_grdmap.h5'
iridiumfn = '/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/lompe/examples/sample_dataset/20120405_iridium.h5'

# set up grid
position = (-90, 65) # lon, lat
orientation = (-1, 2) # east, north
L, W, Lres, Wres = 4200e3, 7000e3, 100.e3, 100.e3 # dimensions and resolution of grid
grid = lompe.cs.CSgrid(lompe.cs.CSprojection(position, orientation), L, W, Lres, Wres, R = 6481.2e3)

# load ampere, supermag, and superdarn data from 2012-05-05
ampere    = pd.read_hdf(iridiumfn)
supermag  = pd.read_hdf(supermagfn)
superdarn = pd.read_hdf(superdarnfn)

# these files contain entire day. Function to select from a smaller time interval is needed:
def prepare_data(t0, t1):
    """ get data from correct time period """
    # prepare ampere
    amp = ampere[(ampere.time >= t0) & (ampere.time <= t1)]
    B = np.vstack((amp.B_e.values, amp.B_n.values, amp.B_r.values))
    coords = np.vstack((amp.lon.values, amp.lat.values, amp.r.values))
    #amp_data = lompe.Data(B * 1e-9, coords, datatype = 'space_mag_fac', scale = 200e-9, error=5e-9)
    #amp_data = lompe.Data(B * 1e-9, coords, datatype = 'space_mag_fac', scale = 200e-9) # Org
    #amp_data = lompe.Data(B * 1e-9, coords, datatype = 'space_mag_fac', scale = 105e-9) # from hist q75
    amp_data = lompe.Data(B * 1e-9, coords, datatype = 'space_mag_fac', components=[0, 1], scale = 105e-9, error=35) # from hist q75 + err

    # prepare supermag
    sm = supermag[t0:t1]
    B = np.vstack((sm.Be.values, sm.Bn.values, sm.Bu.values))
    coords = np.vstack((sm.lon.values, sm.lat.values))
    #sm_data = lompe.Data(B * 1e-9, coords, datatype = 'ground_mag', scale = 100e-9, error=1e-9)
    #sm_data = lompe.Data(B * 1e-9, coords, datatype = 'ground_mag', scale = 100e-9) # Org
    #sm_data = lompe.Data(B * 1e-9, coords, datatype = 'ground_mag', scale = 72e-9) # from hist q75
    sm_data = lompe.Data(B * 1e-9, coords, datatype = 'ground_mag', components=[0, 1], scale = 72e-9, error=1e-9) # from hist q75 + err

    # prepare superdarn
    sd = superdarn.loc[(superdarn.index >= t0) & (superdarn.index <= t1)]
    vlos = sd['vlos'].values
    coords = np.vstack((sd['glon'].values, sd['glat'].values))
    los  = np.vstack((sd['le'].values, sd['ln'].values))
    #sd_data = lompe.Data(vlos, coordinates = coords, LOS = los, datatype = 'convection', scale = 500, error=50)
    #sd_data = lompe.Data(vlos, coordinates = coords, LOS = los, datatype = 'convection', scale = 500) # Org
    #sd_data = lompe.Data(vlos, coordinates = coords, LOS = los, datatype = 'convection', scale = 359) # from hist q75
    sd_data = lompe.Data(vlos, coordinates = coords, LOS = los, datatype = 'convection', scale = 359, error=100) # from hist q75 + err
    
    return amp_data, sm_data, sd_data

# get figures from entire day and save somewhere

# times during entire day
times = pd.date_range('2012-04-05 00:00', '2012-04-05 23:59', freq = '3Min')
DT = timedelta(seconds = 2 * 60) # will select data from +- DT


Kp = 4 # for Hardy conductance model
SH = lambda lon = grid.lon, lat = grid.lat: hardy_EUV(lon, lat, 5, times[0], 'hall'    )
SP = lambda lon = grid.lon, lat = grid.lat: hardy_EUV(lon, lat, 5, times[0], 'pedersen')
model = lompe.Emodel(grid, Hall_Pedersen_conductance = (SH, SP))

#%% Look at data

plt.figure(figsize=(10, 10))
var = np.sqrt(supermag['Be']**2 + supermag['Bn']**2 + supermag['Bu']**2)
plt.hist(var, bins=np.linspace(np.min(var), np.max(var), 100), edgecolor='k', density=True)
plt.title('Supermag : mean {} nT : median {} nT : q75 {} nT'.format(np.round(np.mean(var), 1), np.round(np.median(var), 1), np.round(np.quantile(var, 0.75), 1)))
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/lompe_workshop/supermag_hist.png', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/lompe_workshop/supermag_hist.pdf', format='pdf', bbox_inches='tight')

plt.figure(figsize=(10, 10))
var = copy.deepcopy(superdarn['vlos'])
plt.hist(var, bins=np.linspace(np.min(var), np.max(var), 100), edgecolor='k', density=True)
plt.title('Superdarn : mean {} m/s : median {} m/s : q75 {} m/s'.format(np.round(np.mean(var), 1), np.round(np.median(var), 1), np.round(np.quantile(var, 0.75), 1)))
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/lompe_workshop/superdarn_hist.png', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/lompe_workshop/superdarn_hist.pdf', format='pdf', bbox_inches='tight')

plt.figure(figsize=(10, 10))
var = copy.deepcopy(superdarn['vlos_sd'])
plt.hist(var, bins=np.linspace(np.min(var), np.max(var), 100), edgecolor='k', density=True)
plt.title('Superdarn err : mean {} m/s : median {} m/s : q75 {} m/s'.format(np.round(np.mean(var), 1), np.round(np.median(var), 1), np.round(np.quantile(var, 0.75), 1)))
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/lompe_workshop/superdarn_err_hist.png', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/lompe_workshop/superdarn_err_hist.pdf', format='pdf', bbox_inches='tight')

plt.figure(figsize=(10, 10))
var = np.sqrt(ampere['B_e']**2 + ampere['B_n']**2 + ampere['B_r']**2)
plt.hist(var, bins=np.linspace(np.min(var), np.max(var), 100), edgecolor='k', density=True)
plt.title('Ampere : mean {} nT : median {} nT : q75 {} nT'.format(np.round(np.mean(var), 1), np.round(np.median(var), 1), np.round(np.quantile(var, 0.75), 1)))
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/lompe_workshop/ampere_hist.png', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/lompe_workshop/ampere_hist.pdf', format='pdf', bbox_inches='tight')

plt.figure(figsize=(10, 10))
var = copy.deepcopy(ampere['B_err'])
plt.hist(var, bins=np.linspace(np.min(var), np.max(var), 100), edgecolor='k', density=True)
plt.title('Ampere err : mean {} nT : median {} nT : q75 {} nT'.format(np.round(np.mean(var), 1), np.round(np.median(var), 1), np.round(np.quantile(var, 0.75), 1)))
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/lompe_workshop/ampere_err_hist.png', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/lompe_workshop/ampere_err_hist.pdf', format='pdf', bbox_inches='tight')

plt.close('all')

#%% Independent run

savepath = '/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/lompe_workshop/figs_ind_new_scales/'
try:
    os.mkdir(savepath)
except:
    print('Idiot')

# loop through times and save
#var_ind = {}
for i, t in enumerate(times[1:]):
    print(t)
    
    SH = lambda lon = grid.lon, lat = grid.lat: hardy_EUV(lon, lat, 5, t, 'hall'    )
    SP = lambda lon = grid.lon, lat = grid.lat: hardy_EUV(lon, lat, 5, t, 'pedersen')

    model.clear_model(Hall_Pedersen_conductance = (SH, SP)) # reset
    
    amp_data, sm_data, sd_data = prepare_data(t - DT, t + DT)
    
    model.add_data(amp_data, sm_data, sd_data)
    
    model.run_inversion(l1 = 2, l2 = 0, Cpost=True)
    
    #var_ind['{}'.format(i)] = [model.Cpost_inv, model.m]
    
    savefile = savepath + str(t).replace(' ','_').replace(':','')
    lompe.lompeplot(model, include_data = True, time = t, apex = apex, savekw = {'fname': savefile, 'dpi' : 200})
    plt.close('all')

#%% Estimate process noise
m_all = []
for i in range(len(var_ind)):
    m_all.append(var_ind['{}'.format(i)][1])

m_all = np.array(m_all)
dm_all = np.diff(m_all, axis=0)
Q = np.cov(dm_all, rowvar=False)

Qinv = scipy.linalg.lstsq(Q, np.eye(Q.shape[0]))[0]
#Qinv = np.linalg.lstsq(Q, np.eye(Q.shape[0]))[0]

plt.figure(figsize=(10, 10))
vmax = np.max(np.abs(Q))
plt.imshow(Q, vmin=-vmax, vmax=vmax, cmap='bwr')
plt.colorbar()
plt.title('Process noise estimate')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/lompe_workshop/process_noise_estimate.png', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/lompe_workshop/process_noise_estimate.pdf', format='pdf', bbox_inches='tight')

plt.figure(figsize=(10, 10))
vmax = np.max(np.abs(Qinv))
plt.imshow(Qinv, vmin=-vmax, vmax=vmax, cmap='bwr')
plt.colorbar()
plt.title('Process noise estimate inverted')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/lompe_workshop/process_noise_estimate_inv.png', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/lompe_workshop/process_noise_estimate_inv.pdf', format='pdf', bbox_inches='tight')

#np.save('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/lompe_workshop/Q_estimate.npy', Q)

#%% KF no Q
savepath = '/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/lompe_workshop/figs_KF_l2/'
try:
    os.mkdir(savepath)
except:
    print('Idiot')

# loop through times and save - Taran
for i, t in enumerate(times[1:]):
    print(t)
    
    SH = lambda lon = grid.lon, lat = grid.lat: hardy_EUV(lon, lat, 5, t, 'hall'    )
    SP = lambda lon = grid.lon, lat = grid.lat: hardy_EUV(lon, lat, 5, t, 'pedersen')

    model.clear_model(Hall_Pedersen_conductance = (SH, SP)) # reset
    
    amp_data, sm_data, sd_data = prepare_data(t - DT, t + DT)
    
    model.add_data(amp_data, sm_data, sd_data)
    
    if i == 0:
        model.run_inversion(l1 = 2, l2 = 5e1, Cpost=True)
    else:
        model.run_inversion(Cpost=True, taran=model_old)
    
    model_old = copy.deepcopy(model)
    
    #model_old.Cpost_inv[np.arange(len(model_old.Cpost_inv)).astype(int), np.arange(len(model_old.Cpost_inv)).astype(int)] *= .9
    
    savefile = savepath + str(t).replace(' ','_').replace(':','')
    lompe.lompeplot(model, include_data = True, time = t, apex = apex, savekw = {'fname': savefile, 'dpi' : 200})
    plt.close('all')

#%% KF with Q
Q = np.load('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/lompe_workshop/Q_estimate.npy')

savepath = '/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/lompe_workshop/figs_KF_l2_Sigma1_new_scale_no_Br/'
try:
    os.mkdir(savepath)
except:
    print('Idiot')

# loop through times and save - Taran
var_store = {}
for i, t in enumerate(times[1:]):
    print(t)
    
    SH = lambda lon = grid.lon, lat = grid.lat: hardy_EUV(lon, lat, 5, t, 'hall'    )
    SP = lambda lon = grid.lon, lat = grid.lat: hardy_EUV(lon, lat, 5, t, 'pedersen')

    model.clear_model(Hall_Pedersen_conductance = (SH, SP)) # reset
    
    amp_data, sm_data, sd_data = prepare_data(t - DT, t + DT)
    
    model.add_data(amp_data, sm_data, sd_data)
    
    if i == 0:
        model.run_inversion(l1 = 2, l2 = 5e1, Cpost=True)
    else:
        model.run_inversion(Cpost=True, taran=model_old)
    
    model_old = copy.deepcopy(model)
    
    Rt = scipy.linalg.lstsq(model_old.Cpost_inv, np.eye(model_old.Cpost_inv.shape[0]))[0] + Q
    model_old.Cpost_inv = scipy.linalg.lstsq(Rt, np.eye(Rt.shape[0]))[0]
    
    var_store['{}'.format(i)] = [model.Cpost_inv, model.m]
    
    #model_old.Cpost_inv[np.arange(len(model_old.Cpost_inv)).astype(int), np.arange(len(model_old.Cpost_inv)).astype(int)] *= .9
    
    savefile = savepath + str(t).replace(' ','_').replace(':','')
    lompe.lompeplot(model, include_data = True, time = t, apex = apex, savekw = {'fname': savefile, 'dpi' : 200})
    plt.close('all')

#%% Get reg, but pass on model
savepath = '/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/lompe_workshop/figs_KF_reg_s2_e10/'
try:
    os.mkdir(savepath)
except:
    print('Idiot')

# loop through times and save - Taran
for i, t in enumerate(times[1:]):
    print(t)
    
    SH = lambda lon = grid.lon, lat = grid.lat: hardy_EUV(lon, lat, 5, t, 'hall'    )
    SP = lambda lon = grid.lon, lat = grid.lat: hardy_EUV(lon, lat, 5, t, 'pedersen')

    model.clear_model(Hall_Pedersen_conductance = (SH, SP)) # reset
    
    amp_data, sm_data, sd_data = prepare_data(t - DT, t + DT)
    
    model.add_data(amp_data, sm_data, sd_data)
    
    if i == 0:
        model.run_inversion(l1 = 2, l2 = 0, Cpost=True)
    else:
        model.run_inversion(l1 = 10, l2 = 0, Cpost=True, m_init=m_old)
    
    m_old = copy.deepcopy(model.m)
    
    savefile = savepath + str(t).replace(' ','_').replace(':','')
    lompe.lompeplot(model, include_data = True, time = t, apex = apex, savekw = {'fname': savefile, 'dpi' : 200})
    plt.close('all')


#%%

'''    
# loop through times and save
for t in times[1:]:
    print(t)
    
    SH = lambda lon = grid.lon, lat = grid.lat: hardy_EUV(lon, lat, 5, t, 'hall'    )
    SP = lambda lon = grid.lon, lat = grid.lat: hardy_EUV(lon, lat, 5, t, 'pedersen')

    model.clear_model(Hall_Pedersen_conductance = (SH, SP)) # reset
    
    amp_data, sm_data, sd_data = prepare_data(t - DT, t + DT)
    
    model.add_data(amp_data, sm_data, sd_data)

    model.run_inversion(l1 = 2, l2 = 0)
    
    savefile = savepath + str(t).replace(' ','_').replace(':','')
    lompe.lompeplot(model, include_data = True, time = t, apex = apex, savekw = {'fname': savefile, 'dpi' : 200})
    plt.close('all')
'''


# loop through times and save - Taran
for i, t in enumerate(times[1:]):
    print(t)
    
    SH = lambda lon = grid.lon, lat = grid.lat: hardy_EUV(lon, lat, 5, t, 'hall'    )
    SP = lambda lon = grid.lon, lat = grid.lat: hardy_EUV(lon, lat, 5, t, 'pedersen')

    model.clear_model(Hall_Pedersen_conductance = (SH, SP)) # reset
    
    amp_data, sm_data, sd_data = prepare_data(t - DT, t + DT)
    
    model.add_data(amp_data, sm_data, sd_data)
    
    if i == 0:
        model.run_inversion(l1 = 2, l2 = 0, Cpost=True)
    else:
        model.run_inversion(Cpost=True, taran=model_old)
    
    model_old = copy.deepcopy(model)
    
    model_old.Cpost_inv[np.arange(len(model_old.Cpost_inv)).astype(int), np.arange(len(model_old.Cpost_inv)).astype(int)] *= .9
    
    savefile = savepath + str(t).replace(' ','_').replace(':','')
    lompe.lompeplot(model, include_data = True, time = t, apex = apex, savekw = {'fname': savefile, 'dpi' : 200})
    plt.close('all')


#%%

# loop through times and save - Ind
Cpost_inv_ind = []
for i, t in enumerate(times[1:]):
    print(t)
    
    SH = lambda lon = grid.lon, lat = grid.lat: hardy_EUV(lon, lat, 5, t, 'hall'    )
    SP = lambda lon = grid.lon, lat = grid.lat: hardy_EUV(lon, lat, 5, t, 'pedersen')

    model.clear_model(Hall_Pedersen_conductance = (SH, SP)) # reset
    
    amp_data, sm_data, sd_data = prepare_data(t - DT, t + DT)
    
    model.add_data(amp_data, sm_data, sd_data)
    
    model.run_inversion(l1 = 2, l2 = 0, Cpost=True)
    
    Cpost_inv_ind.append(model.Cpost_inv)
    
    if i == 50:
        break

# loop through times and save - Taran
Cpost_inv_taran = []
for i, t in enumerate(times[1:]):
    print(t)
    
    SH = lambda lon = grid.lon, lat = grid.lat: hardy_EUV(lon, lat, 5, t, 'hall'    )
    SP = lambda lon = grid.lon, lat = grid.lat: hardy_EUV(lon, lat, 5, t, 'pedersen')

    model.clear_model(Hall_Pedersen_conductance = (SH, SP)) # reset
    
    amp_data, sm_data, sd_data = prepare_data(t - DT, t + DT)
    
    model.add_data(amp_data, sm_data, sd_data)
    
    if i == 0:
        model.run_inversion(l1 = 2, l2 = 0, Cpost=True)
    else:
        model.run_inversion(Cpost=True, taran=model_old)
    
    model_old = copy.deepcopy(model)

    Cpost_inv_taran.append(model.Cpost_inv)
    
    if i == 50:
        break

#%%
vmax = np.max([np.max(np.abs(Cpost_inv_taran)), np.max(np.abs(Cpost_inv_ind))])
plt.ioff()
for i in range(50):
    print(i)
    fig, axs = plt.subplots(1, 2, figsize=(15, 9))
    axs[0].imshow(Cpost_inv_ind[i], vmin=-vmax, vmax=vmax, cmap='bwr')
    axs[1].imshow(Cpost_inv_taran[i], vmin=-vmax, vmax=vmax, cmap='bwr')
    axs[0].text(0.5, 1.05, 'Independent : {}'.format(i), va='center', ha='center', transform=axs[0].transAxes)
    axs[1].text(0.5, 1.05, 'Tarantola : {}'.format(i), va='center', ha='center', transform=axs[1].transAxes)
    plt.savefig('/scratch//BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/temp_storage/Cpost/{}.png'.format(i), bbox_inches='tight')
    plt.close('all')
plt.ion()

#%%
'''
    #savefile = savepath + str(t).replace(' ','_').replace(':','')
    #lompe.lompeplot(model, include_data = True, time = t, apex = apex, savekw = {'fname': savefile, 'dpi' : 200})
    #plt.close('all')
    
    plt.figure()
    plt.imshow(model.Cpost_inv)
    plt.colorbar()
    plt.title('Independent {}'.format(i))
    plt.savefig('/scratch//BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/temp_storage/Cpost/{}.png'.format(i), bbox_inches='tight')
    plt.close('all')

'''