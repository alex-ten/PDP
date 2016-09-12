import numpy as np
import matplotlib.pyplot as plt
import time


def main():
    ne = 30

    trainset = load_set('f_XOR.txt')
    net = Network(blueprint(trainset), bias_on=True)
    net.set_hidnet(1,(2,))
    net.connect()
    net.connections[0].weights = np.array([[0.432171, 0.448781], [-0.038413, 0.036489]])
    net.connections[1].weights = np.array([[0.272080, 0.081714]])
    net.biases[0].weights = np.array([-0.27659, -0.40250])
    net.biases[1].weights = np.array([0.27930])
    #net.summary()
    net.train(trainset,numepochs=ne,lrate=0.5,lmode='sim',decay=0.0, momentum=0.9, permute=False)

class Network(object):
    def __init__(self, blueprint: object, bias_on: object = False) -> object:
        self.connections = []
        self.layers = []
        self.hidnet = []
        self.inp = Layer(blueprint[0], 'input')
        self.outp = Layer(blueprint[1], 'output')
        self.trinp = np.zeros(blueprint[0]) # training input pattern(s)
        self.target = np.zeros(blueprint[1]) # target output pattern(s)
        self.flow = [self.inp, self.outp]
        self.bias_on = bias_on
        self.biases = []
        self.tss = []
        self.tce = []
        self.stats_data = {'TSS':[],'TCE':[], 'ACTS':[], 'PROJ':[], 'BIAS':[], 'DELTA':[], 'WED':[], 'WEDMOM':[]}
        self.hidact = [] #TODO delete

    def summary(self):
        for layer in self.flow:
            print('---' + str(layer) + '---')
            try:
                print('1. Net input  ...', layer.netinp)
                print('2. Activation ...', layer.activation)
                print('3. Delta ........', layer.delta)
            except AttributeError: pass
            print('4. Sender .......', layer.sender)
            print('5. Receiver .....', layer.receiver)
            print('6. Afferent .....', layer.afferent)
            try:
                print('   Weights:\n', layer.afferent.weights)
            except AttributeError: pass
            print('7. Efferent .....', layer.efferent)
            try:
                print('   Weights:\n', layer.efferent.weights)
            except AttributeError: pass
            print('8. Bias .........', layer.bias)
            print('\n')

    def set_hidnet(self, *args):
        # Construct hidden layers within Network
        # By default, build 1 hidden layer with the same number of units as in the input layer
        # A single integer argument denotes the number of hidden layers and the function passes control
        # to user to input the number of hidden units in each layer
        # An integer with a tuple of length=integer, builds a hidden net that has integer hidden layers,
        # with each element of the tuple determining the number of units in layer integer
        if not args:
            self.hidnet.insert(0, Layer(self.inp.numunits, 'hidden'))
            self.outp.eyed = 2
        elif len(args) == 1:
            nhl = args[0]
            print('Set up a {}-layer hidden network'.format(nhl))
            for layer in range(0, nhl):
                nhu = int(input('Enter the number of units in hidden layer {}: '.format(layer)))
                self.hidnet.insert(layer, Layer(nhu, 'hidden'))
                self.outp.eyed = nhl + 1
        elif len(args) == 2:
            nhl = args[0]  # number of hidden layers
            nhu = args[1]  # number of hidden units
            if any(type(n) != int for n in nhu) or type(nhl) != int:
                raise ValueError('All arguments must integers')
            elif nhl != len(nhu):
                raise ValueError('Need a set of units for each '
                                 'layer and have one layer per each set of '
                                 'units. Got {} layers and {} sets of units'.format(nhl, len(nhu)))
            elif any(n <= 0 for n in nhu):
                raise ValueError("Can't construct hidden layer with "
                                 "number of units <= 0 "
                                 "Check the tuple argument")
            else:
                for layer in range(0, nhl):  # for each hidden layer
                    self.hidnet.insert(layer, Layer(nhu[layer], 'hidden'))
            self.outp.eyed = nhl + 1
        if len(self.hidnet) >= 1: # Assign network id to each hidden layer (starting from id: 1)
            for i,layer in enumerate(self.hidnet): layer.eyed = i + 1

    def set_flow(self):
        if self.hidnet==[]:
            self.inp.receiver = self.outp  # Set receiver attribute for input Layer
            self.inp.efferent = self.connections[0]
            self.outp.sender = self.inp  # Set sender attribute for output Layer
            self.outp.afferent = self.connections[0]
        else:
            self.inp.receiver = self.hidnet[0] # Set receiver attribute for input Layer
            self.inp.efferent = self.connections[0]
            self.outp.sender = self.hidnet[-1] # Set sender attribute for output Layer
            self.outp.afferent = self.connections[-1]
            templist = [self.inp] + self.hidnet + [self.outp]
            for i, layer in enumerate(self.hidnet): # Set sender / receiver attribute for objects in self.hidnet
                layer.sender = templist[i]
                layer.receiver = templist[i+2]
                layer.afferent = self.connections[i]
                layer.efferent = self.connections[i+1]
                layer.bias.layer = layer
            self.outp.bias.layer = self.outp
            self.flow = [self.inp] + self.hidnet + [self.outp]

    def connect(self, wrange=0.5, constraint='np'):
        FROM = [self.inp] + self.hidnet # List containig input layer and all hidden layers
        TO = self.hidnet + [self.outp] # List of all hidden layers and output layer
        pairs = merge(FROM,TO)
        for member in pairs:
            rows = member[1].numunits
            cols = member[0].numunits
            values = np.random.rand(rows, cols)
            link = (member[1].eyed, member[0].eyed)
            sender, receiver = member[0], member[1]
            W = Projection(values, sender, receiver, link)
            B = np.random.rand(rows)
            if constraint == 'p':
                W.weights = W.weights * wrange
                member[1].bias.set_weights(B * wrange)
            elif constraint == 'n':
                W.weights = W.weights * -wrange
                member[1].bias.set_weights(B * -wrange)
            else:
                W.weights = ((W.weights * 2) - 1) * wrange/2
                member[1].bias.set_weights(((B * 2) - 1) * wrange/2)
            self.connections.insert(member[0].eyed, W)
            self.biases.insert(member[1].eyed, member[1].bias)
        self.set_flow()

    def feed(self, pattern_set, use='fp'):
        self.inp.reset()
        self.outp.reset()
        if use=='train':  # Insert training set
            for key, subdict in pattern_set.items():
                if key=='input':
                    self.trinp = np.array([pattern for pattern in subdict.values()])
                else: self.target = np.array([pattern for pattern in subdict.values()])
        elif use=='test':
            for key, subdict in pattern_set.items():
                if key == 'input':
                    self.inset = np.array([pattern for pattern in subdict.values()])
                else: self.outset = np.array([pattern for pattern in subdict.values()])
        elif use=='fp':
            self.inp.activation = np.array([pattern for pattern in pattern_set['input'].values()])

    def f_prop(self):
            for layer in self.flow:
                if layer.kind=='input': layer.activation = self.inp.activation
                else:
                    if self.bias_on:
                        layer.excite(layer.sender.activation, layer.afferent.weights, bias=True)
                    else:
                        layer.excite(layer.sender.activation, layer.afferent.weights)
            return self.outp.activation

    def b_prop(self):
        error = self.outp.compute_error(self.target, self.outp.activation, funct='SE', der=True)
        self.outp.bias.delta = error
        for layer in reversed(self.hidnet):
            if layer.receiver.kind == 'output':
                layer.compute_delta(error, layer.efferent.weights)
                layer.bias.delta = layer.delta
            else:
                layer.compute_delta(layer.receiver.delta, layer.efferent.weights)
                layer.bias.delta = layer.delta

    def compute_stats(self, disp=False):
        SEs = squerr(self.target, self.outp.activation)
        CEs = cross_entropy(self.target, self.outp.activation)
        try:
            pss = np.sum(SEs, axis=1)
            pce = np.sum(CEs, axis=1)
        except ValueError:
            pss = np.sum(SEs, axis=0)
            pce = np.sum(CEs, axis=0)
        self.tss.append(pss)
        self.tce.append(pce)
        if disp:
            print('    PSS: ', pss)
            print('    PCE: ', pce)

    def store_values(self, store='acts'):
        if store=='acts':
            self.stats_data['ACTS'].append(self.hidnet[0].activation[0])
            self.stats_data['ACTS'].append(self.hidnet[0].activation[1])
            self.stats_data['ACTS'].append(self.outp.activation[:])
            self.stats_data['DELTA'].append(self.hidnet[0].delta[0])
            self.stats_data['DELTA'].append(self.hidnet[0].delta[1])
            self.stats_data['DELTA'].append(self.outp.delta[:])
        elif store=='weds':
            self.stats_data['WED'].append(self.connections[0].wed[0, 0])
            self.stats_data['WED'].append(self.connections[0].wed[0, 1])
            self.stats_data['WED'].append(self.connections[0].wed[1, 0])
            self.stats_data['WED'].append(self.connections[0].wed[1, 1])
            self.stats_data['WED'].append(self.connections[1].wed[0, 0])
            self.stats_data['WED'].append(self.connections[1].wed[0, 1])
            self.stats_data['WED'].append(self.biases[0].wed[0])
            self.stats_data['WED'].append(self.biases[0].wed[1])
            self.stats_data['WED'].append(self.biases[1].wed[0])
        elif store == 'mweds':
            self.stats_data['WEDMOM'].append(self.connections[0].momentum[0, 0]*0.9)
            self.stats_data['WEDMOM'].append(self.connections[0].momentum[0, 1]*0.9)
            self.stats_data['WEDMOM'].append(self.connections[0].momentum[1, 0]*0.9)
            self.stats_data['WEDMOM'].append(self.connections[0].momentum[1, 1]*0.9)
            self.stats_data['WEDMOM'].append(self.connections[1].momentum[0, 0]*0.9)
            self.stats_data['WEDMOM'].append(self.connections[1].momentum[0, 1]*0.9)
            self.stats_data['WEDMOM'].append(self.biases[0].momentum[0]*0.9)
            self.stats_data['WEDMOM'].append(self.biases[0].momentum[1]*0.9)
            self.stats_data['WEDMOM'].append(self.biases[1].momentum[0]*0.9)
        elif store=='weights':
            self.stats_data['PROJ'].append(self.connections[0].weights[0,0])
            self.stats_data['PROJ'].append(self.connections[0].weights[0,1])
            self.stats_data['PROJ'].append(self.connections[0].weights[1,0])
            self.stats_data['PROJ'].append(self.connections[0].weights[1,1])
            self.stats_data['PROJ'].append(self.connections[1].weights[0,0])
            self.stats_data['PROJ'].append(self.connections[1].weights[0,1])
            self.stats_data['BIAS'].append(self.biases[0].weights[0])
            self.stats_data['BIAS'].append(self.biases[0].weights[1])
            self.stats_data['BIAS'].append(self.biases[1].weights[0])

    def sum_stats(self):
        self.tss = sum(self.tss)
        rtss = self.tss
        self.stats_data['TSS'].append(self.tss)
        self.tce = sum(self.tce)
        rtce = self.tce
        self.stats_data['TCE'].append(self.tce)
        print('    TSS: ', self.tss)
        # print('    TCE: ', self.tce) #TODO untag
        self.tss, self.tce = [], []
        return rtss, rtce

    def compute_weds(self):
        for projection in self.connections:
            projection.compute_wed()
        for bias_unit in self.biases:
            bias_unit.compute_wed()

    def update_network(self, e, a, o):
        for projection in self.connections:
            projection.onUpdate(epsilon=e, alpha=a, omega=o)
        for bias_unit in self.biases:
            bias_unit.update_weights(epsilon=e, alpha=a, omega=o)

    def train(self, trainset, numepochs=10, lrate=0.5, lmode='seq', decay=0.0, momentum=0.0, permute=False):
        run = input('Run the training routine automatically (yes,no)?: ')
        if run == 'yes': brk = int(input('Set break frequency: '))
        if lmode == 'hyb':
            z = 1
            step = int(input('Set update frequency: '))
        if (numepochs > 10) & (numepochs <= 100): rep = 10
        elif (numepochs > 100) & (numepochs <= 1000): rep = 100
        else: rep = 1000
        epoch = 0
        temp=[] #todo delete
        t0 = time.time()
        while epoch < numepochs:
            self.feed(trainset, use='train')
            if permute: uni_shuffle(self.trinp, self.target)
            pairs = merge(self.trinp, self.target)
            #print('±±±'*9,'Training epoch {}'.format(epoch),'±±±'*9)
            # if run == 'yes':
            #     if (epoch) % brk == 0:
            #         self.summary()
            #         input('Press ruturn to begin the training epoch')
            for pair in pairs:
                # if run == 'no': input('> PRESS RETURN TO TRAIN WITH PATTERN PAIR: {} -> {}'. format(pair[0], pair[1]))
                # else: print('> TRAINING WITH PATTERN PAIR: {} -> {}'. format(pair[0], pair[1]))
                self.inp.activation = pair[0]
                self.target = pair[1]
                self.f_prop()
                self.compute_stats()
                self.b_prop()
                #self.store_values() # STORE ACTIVATION VALUES
                if run == 'no': self.summary()
                elif run == 'yes':
                    if epoch % rep == 0: self.summary()
                self.compute_weds()
                if lmode == 'seq':
                    self.update_network(lrate, momentum, decay) # positional arguments (lrate,momentum,decay)
                    if run=='yes':
                        if epoch % rep == 0: self.summary()
                elif lmode == 'hyb':
                    if z % step == 0:
                        self.update_network(lrate, momentum, decay) # positional arguments (lrate,momentum,decay)
                        if run == 'yes':
                            if epoch % rep == 0: self.summary()
                    z += 1
            if lmode == 'sim':
                self.store_values(store='weds') # STORE WEDS
                self.update_network(lrate, momentum, decay) # positional arguments (lrate,momentum,decay)
                self.store_values(store='mweds')  # STORE WEDS
                if run == 'yes':
                    if epoch % rep == 0: self.summary()
            self.sum_stats()
            self.store_values(store='weights')
            self.hidact.append(temp) #TODO delete
            temp = [] #TODO delete
            epoch += 1
        t1 = time.time()
        print('TIME: {}'.format(t1-t0))

    def test(self, testset):
        print('> <'*10,'Testing','> <'*10)
        self.feed(testset, use='test')
        numsets = np.shape(self.inset)[0]
        pairs = merge(self.inset, self.outset)
        for pair in pairs:
            self.inp.activation = pair[0]
            self.target = pair[1]
            self.f_prop()
            error = (self.target - self.outp.activation)
            print('{} --> {}, error: {}'.format(self.inp.activation, np.around(self.outp.activation,2), np.around(error,2)))
            error = (self.target - self.outp.activation)**2

        print('  INSIDES:')
        for b in self.biases:
            print(b)
        for w in self.connections:
            print(w)
            print(w.weights)
        print('Mean squared error: {}'.format(error/numsets))

class Layer(object):
    def __init__(self, layer, kind):
        self.netinp = np.zeros(layer)
        self.activation = np.zeros(layer)
        self.delta = np.zeros(layer)
        self.sender = None
        self.receiver = None
        self.afferent = None
        self.efferent = None
        self.numunits = layer
        self.kind = kind
        self.eyed = 0
        self.bias = Bias(layer)
        self.act_funct = 'logistic'
    def __str__(self):
        return "<Layer object, kind: " + self.kind + ', level: {}, units: {}>'.format(self.eyed, self.numunits)

    def excite(self, X, w, bias=False):
        if bias: self.netinp = X.dot(w.T) + self.bias.weights
        else: self.netinp = X.dot(w.T)
        if self.act_funct=='logistic':
            self.activation = logact(self.netinp)
        else:
            self.activation = self.netinp

    def compute_error(self, t, a, funct='SE', der=True): # der was False
        if self.act_funct=='logistic':
            if funct=='SE': self.delta =  squerr(t, a, deriv=der) * logact(self.activation, deriv=der)
            elif funct=='CE': self.delta =  squerr(t, a, deriv=der)
        else:
            self.delta =  squerr(t, a, deriv=der)
        return self.delta

    def compute_delta(self, d, w):
        if self.act_funct == 'logistic':
            self.delta = logact(self.activation, deriv=True) * d.dot(w)
        else:
            self.delta = d.dot(w)

    def reset(self):
        self.netinp = self.netinp * 0
        self.activation = self.activation * 0
        self.numunits += -1
        self.delta * 0

class Projection(object):
    def __init__(self, matrix, sender, receiver, link):
        self.weights = matrix
        self.momentum = np.zeros(np.shape(matrix))
        self.link = link # 2-tuple containing indices (eyeds) of layers linked by the projection. E.g. (to,from)
        self.sender = sender
        self.receiver = receiver
        self.wed = np.zeros(np.shape(matrix))
    def __str__(self):
        return '<Projection ' + str(self.link[1]) + ' ~> ' + str(self.link[0])+'>'

    def compute_wed(self):
        self.wed += np.outer(self.receiver.delta, self.sender.activation)

    def update(self, epsilon, alpha, omega):
        decay = omega * self.weights
        momentum = alpha * self.momentum
        self.weights += epsilon * self.wed - decay + momentum
        self.momentum = epsilon * self.wed - decay + momentum
        self.wed = self.wed * 0

class Bias(object):
    def __init__(self, numunits):
        self.weights = np.empty((0, numunits))
        self.layer = None
        self.delta = np.zeros(numunits)
        self.wed = np.zeros(numunits)
        self.momentum = np.zeros(numunits)

    def __str__(self):
        return '<Bias object, weights ' + str(self.weights)+ '>'

    def set_weights(self, B):
        self.weights = B

    def compute_wed(self):
        self.wed += self.delta * 1

    def update_weights(self, epsilon, alpha, omega):
        decay = omega * self.weights
        momentum = alpha * self.momentum
        self.weights += epsilon * self.wed - decay + momentum
        self.momentum = epsilon * self.wed - decay + momentum
        self.wed = self.wed * 0


def load_set(filename):
    pattern_set = {}
    for numline, line in enumerate(open(filename)):
        if len(line) <= 1: continue
        pattern_set.setdefault('input',{})
        pattern_set.setdefault('output',{})
        if numline == 0:
            i_ind = line.find('input')
            o_ind = line.rfind('output')
        else:
            ilist = [int(x) for x in line[i_ind:o_ind].split()]
            olist = [int(x) for x in line[o_ind:].split()]
            pattern_set['input'][numline] = np.array(ilist)
            pattern_set['output'][numline] = np.array(olist)
    return pattern_set

def display_set(s):
    for l, sub in sorted(s.items()):
        for k, v in s[l].items():
            print(l + ' {}: {}'.format(k, v))
    print('\n')

def blueprint(s): # Return number of units in inslot and outmod
    inslot_size = np.size(s['input'][1])
    outmod_size = np.size(s['output'][1])
    return (inslot_size, outmod_size)

def uni_shuffle(a, b): # Shuffle arrays a and b in unison
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def merge(a, b): # Merges two iterables into a list,  sothat [a[i],b[i]] is followed by [a[i+1],b[i+1]]
    outlist = []
    for i, eyetem in enumerate(a):
        outlist.insert(i,[eyetem,b[i]])
    return outlist

def logact(x,deriv=False): # Logistic activation function (sigmoid) and its derivative
    if deriv:
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))

def cross_entropy(t, a, deriv=False): # Cross-entropy error function and its derivative
    if deriv:
        return (t/a) + (1-t)/(1-a)
    return t * np.log(a) + (1 - t) * np.log(1 - a)

def squerr(t, a, deriv=False): # Square difference function and its derivative
    if deriv:
        return t - a
    return (t - a)**2

def listlist(a):
    for n,i in enumerate(a):
        print('Item {} from list:\n'.format(n),i)

def plot_stats(N, numepochs, weds=False, mweds=False):
    if weds:
        plt.figure(3)
        w1_11 = N.stats_data['WED'][0::9]
        w1_12 = N.stats_data['WED'][1::9]
        w1_21 = N.stats_data['WED'][2::9]
        w1_22 = N.stats_data['WED'][3::9]
        w2_11 = N.stats_data['WED'][4::9]
        w2_12 = N.stats_data['WED'][5::9]
        b1_11 = N.stats_data['WED'][6::9]
        b1_12 = N.stats_data['WED'][7::9]
        b2_11 = N.stats_data['WED'][8::9]

        plt.subplot(2, 1, 1)
        plt.title('Weights over {} epochs'.format(numepochs))
        plt.grid(True)
        plt.ylabel('Weight')
        plt.plot(w1_11, color='mediumpurple', linewidth=2, label='wj_11')
        plt.plot(w1_21, color='palegreen', linewidth=2, label='wj_21')
        plt.plot(w1_12, color='darkblue', linewidth=2, label='wj_12')
        plt.plot(w1_22, color='forestgreen', linewidth=2, label='wj_22')
        plt.plot(w2_11, color='darkred', linewidth=2, linestyle='--', label='wi_11')
        plt.plot(w2_12, color='salmon', linewidth=2, linestyle='--', label='wi_12')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)

        plt.subplot(2, 1, 2)
        plt.title('Bias weights over {} epochs'.format(numepochs))
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Bias')
        plt.plot(b1_11, color='black', linewidth=2, label='bj_1')
        plt.plot(b1_12, color='lightslategray', linewidth=2, label='bj_2')
        plt.plot(b2_11, color='silver', linewidth=2, label='bi_1')
        plt.subplots_adjust(hspace=0.6)
    elif mweds:
        plt.figure(4)
        w1_11 = N.stats_data['WEDMOM'][0::9]
        w1_12 = N.stats_data['WEDMOM'][1::9]
        w1_21 = N.stats_data['WEDMOM'][2::9]
        w1_22 = N.stats_data['WEDMOM'][3::9]
        w2_11 = N.stats_data['WEDMOM'][4::9]
        w2_12 = N.stats_data['WEDMOM'][5::9]
        b1_11 = N.stats_data['WEDMOM'][6::9]
        b1_12 = N.stats_data['WEDMOM'][7::9]
        b2_11 = N.stats_data['WEDMOM'][8::9]

        plt.subplot(2, 1, 1)
        plt.title('Weights over {} epochs'.format(numepochs))
        plt.grid(True)
        plt.ylabel('Weight')
        plt.plot(w1_11, color='mediumpurple', linewidth=2, label='wj_11')
        plt.plot(w1_21, color='palegreen', linewidth=2, label='wj_21')
        plt.plot(w1_12, color='darkblue', linewidth=2, label='wj_12')
        plt.plot(w1_22, color='forestgreen', linewidth=2, label='wj_22')
        plt.plot(w2_11, color='darkred', linewidth=2, linestyle='--', label='wi_11')
        plt.plot(w2_12, color='salmon', linewidth=2, linestyle='--', label='wi_12')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)

        plt.subplot(2, 1, 2)
        plt.title('Bias weights over {} epochs'.format(numepochs))
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Bias')
        plt.plot(b1_11, color='black', linewidth=2, label='bj_1')
        plt.plot(b1_12, color='lightslategray', linewidth=2, label='bj_2')
        plt.plot(b2_11, color='silver', linewidth=2, label='bi_1')
        plt.subplots_adjust(hspace=0.6)
    else:
        plt.figure(1)
        tss = N.stats_data['TSS']
        # tce = N.stats_data['TCE'] #TODO Untag to plot TCE
        w1_11 = N.stats_data['PROJ'][0::6]
        w1_12 = N.stats_data['PROJ'][1::6]
        w1_21 = N.stats_data['PROJ'][2::6]
        w1_22 = N.stats_data['PROJ'][3::6]
        w2_11 = N.stats_data['PROJ'][4::6]
        w2_12 = N.stats_data['PROJ'][5::6]
        b1_11 = N.stats_data['BIAS'][0::3]
        b1_12 = N.stats_data['BIAS'][1::3]
        b2_11 = N.stats_data['BIAS'][2::3]


        plt.subplot(3, 1, 1)
        plt.title('TSS, weights, and biases over {} epochs'.format(numepochs))
        plt.grid(True)
        plt.ylabel('TSS')
        plt.plot(tss, color='orange', linewidth=2)
        #plt.plot(tce, color='blue', linewidth=2) #TODO Untag to plot TCE
        plt.legend(['TSS'], loc='upper right')
        #plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8) #TODO Untag to plot TCE

        plt.subplot(3, 1, 2)
        plt.title('Weights over {} epochs'.format(numepochs))
        plt.grid(True)
        plt.ylabel('Weight')
        plt.plot(w1_11, color = 'mediumpurple', linewidth=2, label='wj_11')
        plt.plot(w1_21, color = 'palegreen', linewidth = 2, label='wj_21')
        plt.plot(w1_12, color = 'darkblue', linewidth = 2, label='wj_12')
        plt.plot(w1_22, color = 'forestgreen', linewidth = 2, label='wj_22')
        plt.plot(w2_11, color = 'darkred', linewidth = 2, linestyle='--', label='wi_11')
        plt.plot(w2_12, color = 'salmon', linewidth = 2, linestyle='--', label='wi_12')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)

        plt.subplot(3, 1, 3)
        plt.title('Bias weights over {} epochs'.format(numepochs))
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Bias')
        plt.plot(b1_11, color='black', linewidth=2,label='bj_1')
        plt.plot(b1_12, color='lightslategray', linewidth=2,label='bj_2')
        plt.plot(b2_11, color='silver', linewidth=2,label='bi_1')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
        plt.subplots_adjust(hspace=0.6)

def plot_acts(N, numepochs):
    plt.figure(2)
    x = range(numepochs)
    ms = 80
    h1_00 = N.stats_data['ACTS'][0::12]
    h1_01 = N.stats_data['ACTS'][3::12]
    h1_10 = N.stats_data['ACTS'][6::12]
    h1_11 = N.stats_data['ACTS'][9::12]
    h2_00 = N.stats_data['ACTS'][1::12]
    h2_01 = N.stats_data['ACTS'][4::12]
    h2_10 = N.stats_data['ACTS'][7::12]
    h2_11 = N.stats_data['ACTS'][10::12]
    a1_00 = N.stats_data['ACTS'][2::12]
    a1_01 = N.stats_data['ACTS'][5::12]
    a1_10 = N.stats_data['ACTS'][8::12]
    a1_11 = N.stats_data['ACTS'][11::12]

    d1_00 = np.array(N.stats_data['DELTA'][0::12])
    d1_01 = np.array(N.stats_data['DELTA'][3::12])
    d1_10 = np.array(N.stats_data['DELTA'][6::12])
    d1_11 = np.array(N.stats_data['DELTA'][9::12])
    d1_s = d1_00 + d1_01 + d1_10 + d1_11
    d2_00 = np.array(N.stats_data['DELTA'][1::12])
    d2_01 = np.array(N.stats_data['DELTA'][4::12])
    d2_10 = np.array(N.stats_data['DELTA'][7::12])
    d2_11 = np.array(N.stats_data['DELTA'][10::12])
    d2_s = d2_00 + d2_01 + d2_10 + d2_11
    d3_00 = np.array(N.stats_data['DELTA'][2::12])
    d3_01 = np.array(N.stats_data['DELTA'][5::12])
    d3_10 = np.array(N.stats_data['DELTA'][8::12])
    d3_11 = np.array(N.stats_data['DELTA'][11::12])
    d3_s = d3_00 + d3_01 + d3_10 + d3_11

    plt.subplot(3, 2, 1)
    plt.title('Activation of j1 over {} epochs'.format(numepochs))
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.xlim(0,numepochs)
    plt.ylabel('Activation')
    plt.ylim(0, 1)
    plt.scatter(x, h1_00, marker='+', label='00', color='red', s=ms)
    plt.scatter(x, h1_01, marker='x', label='01', color='blue', s=ms)
    plt.scatter(x, h1_10, marker='4', label='10', color='deeppink', s=ms)
    plt.scatter(x, h1_11, marker='|', label='11', color='green', s=ms)
    plt.subplots_adjust(hspace=0.6)

    plt.subplot(3, 2, 2)
    plt.title('Delta of j1 over {} epochs'.format(numepochs))
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.xlim(0, numepochs)
    plt.ylabel('Delta')
    plt.scatter(x, d1_00, marker='+', label='00', color='red', s=ms)
    plt.scatter(x, d1_01, marker='x', label='01', color='blue', s=ms)
    plt.scatter(x, d1_10, marker='4', label='10', color='deeppink', s=ms)
    plt.scatter(x, d1_11, marker='|', label='11', color='green', s=ms)
    plt.plot(d1_s, color='black', lw=1.5)
    plt.legend(bbox_to_anchor=(1, 0.5), loc='center left', ncol=1)
    plt.subplots_adjust(hspace=0.6)

    plt.subplot(3, 2, 3)
    plt.title('Activation of j2 over {} epochs'.format(numepochs))
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.xlim(0, numepochs)
    plt.ylabel('Activation')
    plt.ylim(0, 1)
    plt.scatter(x, h2_00, marker='+', label='00', color='red', s=ms)
    plt.scatter(x, h2_01, marker='x', label='01', color='blue', s=ms)
    plt.scatter(x, h2_10, marker='4', label='10', color='deeppink', s=ms)
    plt.scatter(x, h2_11, marker='|', label='11', color='green', s=ms)
    plt.subplots_adjust(hspace=0.6)

    plt.subplot(3, 2, 4)
    plt.title('Delta of j2 over {} epochs'.format(numepochs))
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.xlim(0, numepochs)
    plt.ylabel('Delta')
    plt.scatter(x, d2_00, marker='+', label='00', color='red', s=ms)
    plt.scatter(x, d2_01, marker='x', label='01', color='blue', s=ms)
    plt.scatter(x, d2_10, marker='4', label='10', color='deeppink', s=ms)
    plt.scatter(x, d2_11, marker='|', label='11', color='green', s=ms)
    plt.plot(d2_s, color='black', lw=1.5)
    plt.legend(bbox_to_anchor=(1, 0.5), loc='center left', ncol=1)
    plt.subplots_adjust(hspace=0.6)

    plt.subplot(3, 2, 5)
    plt.title('Activation of i over {} epochs'.format(numepochs))
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.xlim(0, numepochs)
    plt.ylabel('Activation')
    plt.ylim(0, 1)
    plt.scatter(x, a1_00, marker='+', label='00', color='red', s=ms)
    plt.scatter(x, a1_01, marker='x', label='01', color='blue', s=ms)
    plt.scatter(x, a1_10, marker='4', label='10', color='deeppink', s=ms)
    plt.scatter(x, a1_11, marker='|', label='11', color='green', s=ms)
    plt.subplots_adjust(hspace=0.6)

    plt.subplot(3, 2, 6)
    plt.title('Delta of i over {} epochs'.format(numepochs))
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.xlim(0, numepochs)
    plt.ylabel('Delta')
    plt.scatter(x, d3_00, marker='+', label='00', color='red', s=ms)
    plt.scatter(x, d3_01, marker='x', label='01', color='blue', s=ms)
    plt.scatter(x, d3_10, marker='4', label='10', color='deeppink', s=ms)
    plt.scatter(x, d3_11, marker='|', label='11', color='green', s=ms)
    plt.plot(d3_s, color='black', lw=1.5)
    plt.legend(bbox_to_anchor=(1, 0.5), loc='center left', ncol=1)
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.90, top=0.95, hspace=0.30, wspace=0.20)


if __name__ == "__main__": main()