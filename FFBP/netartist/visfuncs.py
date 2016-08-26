import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def fp_axes(field, figure):
    sub = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=field, hspace=0.01)
    rd = {}
# weights
    axW = plt.Subplot(figure, sub[0, 0])
    rd['W'] = axW
    axW.set_title(r'$w$', fontsize=11, position=(0.5,1.1))
    figure.add_subplot(axW)
# input
    axInp = plt.Subplot(figure, sub[1, 0])
    rd['inp'] = axInp
    axInp.set_xlabel(r'$input$', fontsize=11)
    axInp.set(adjustable='box-forced')
    figure.add_subplot(axInp)
# biases
    axB = plt.Subplot(figure, sub[0, 1])
    rd['b'] = axB
    axB.set_title(r'$b$', fontsize=11, position=(0.5,1.1))
    figure.add_subplot(axB)
# net input
    axNet = plt.Subplot(figure, sub[0, 2])
    rd['netinp'] = axNet
    axNet.set_title(r'$net$', fontsize=11, position=(0.5,1.1))
    figure.add_subplot(axNet)
# activations
    axAct = plt.Subplot(figure, sub[0, 3])
    rd['activations'] = axAct
    axAct.set_title(r'$act$', fontsize=11, position=(0.5,1.1))
    figure.add_subplot(axAct)
    return rd

def bp_axes(field, figure):
    sub = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=field)
    rd = {}

    axdW = plt.Subplot(figure, sub[0, 0])
    rd['ded_W'] = axdW
    axdW.set_title(r'$\frac{\partial E}{\partial w}$', fontsize=14, position=(0.5,1.1))
    figure.add_subplot(axdW)

    axdB = plt.Subplot(figure, sub[0, 1])
    rd['ded_b'] = axdB
    axdB.set_title(r'$\frac{\partial E}{\partial bias}$', fontsize=14, position=(0.5,1.1))
    figure.add_subplot(axdB)

    axdNet = plt.Subplot(figure, sub[0, 2])
    rd['ded_netinp'] = axdNet
    axdNet.set_title(r'$\frac{\partial E}{\partial net}$', fontsize=14, position=(0.5,1.1))
    figure.add_subplot(axdNet)

    axdAct = plt.Subplot(figure, sub[0, 3])
    rd['ded_activations'] = axdAct
    axdAct.set_title(r'$\frac{\partial E}{\partial act}$', fontsize=14, position=(0.5,1.1))
    figure.add_subplot(axdAct)
    return rd

def snap_viewer(path):
    np.set_printoptions(precision=3, suppress=True, linewidth=200)

    with open(path, 'rb') as opened_file:
        old_snap = pickle.load(opened_file)

    print('Error:')
    print(old_snap['error'], end='\n\n')

    for keys, subdicts in old_snap.items():
        if keys != 'error':
            print('>>> ' + keys + ':')
            for k, v in subdicts.items():
                print('    ' + k + ':')
                print(v)
            print('===' * 50, end='\n\n')

def rollup(strip):
    rows = strip[1]
    cols = strip[2]
    return np.reshape(strip[3:], (rows,cols))


