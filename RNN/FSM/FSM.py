from random import randint
import pickle

State = type('State', (object,), {})
Transition = type('Transition', (object,), {})

class IntState(State):
    def __init__(self, *args, **kwargs):
        args = list(args)
        self.i = args.pop(0)
        super(IntState, self).__init__(*args, **kwargs)
    def Exec(self):
        print('state '+str(self.i))


class SimpleTransition(Transition):
    def __init__(self, *args, **kwargs):
        args = list(args)
        self.to = args.pop()
        self.from_ = args.pop()
        self.sid = kwargs.pop('sid')
        super(SimpleTransition, self).__init__()
    def Exec(self):
        return self.to


class FSM():
    def __init__(self, states, transitions):
        self.start = 'B '
        self.end = '\n'
        self.states = states
        self.transitions = transitions
        self.num_states = len(states)
        self.in_state = states[0]
        self.in_trans = None
        self.state_history = []
        self.trans_history = []
        self.composition = self.start
        self.trans_set = list(set([t.sid for t in transitions] + [self.start, self.end]))

    def transition(self, record = False):
        fork = [x for x in self.transitions if x.from_ == self.in_state]
        choice = fork[randint(0,1)]
        self.composition += choice.sid
        if record:
            self.state_history.append(self.in_state)
            self.trans_history.append(choice.sid)
        self.in_state = choice.Exec()


    def utter(self, max_units):
        self.reset()
        for step in range(max_units):
            self.transition()
            if self.in_state == self.states[-1]: break
        self.composition += '\n'
        print(self.composition)
        return self.composition

    def chatter(self, num_utterances, max_chars):
        chatter_log = []
        for i in range(num_utterances):
            chatter_log.append(self.utter(max_chars))
        chatter_log.append(self.trans_set)
        return chatter_log

    def reset(self):
        self.in_state = self.states[0]
        self.composition = 'B '


def pickle_random_chatter(log, path):
    pickle.dump(log, open(path, 'wb'))

def save_txt(log, path):
    with open(path+'new_file', 'w') as f:
        for line in log:
            f.write(str(line))

def make_states(l):
    return [IntState(s) for s in l]


def make_transitions(td):
    return [SimpleTransition(k[0],k[1], sid = v) for k,v in td.items()]


def main():
    l = list(range(0,6))
    states = make_states(l)
    td = {(states[0],states[1]): 'T ',
          (states[2],states[2]): 'T ',
          (states[1],states[1]): 'S ',
          (states[3],states[5]): 'S ',
          (states[1],states[3]): 'X ',
          (states[3],states[2]): 'X ',
          (states[2],states[4]): 'V ',
          (states[4],states[5]): 'V ',
          (states[4],states[3]): 'P ',
          (states[0],states[2]): 'P ',
          }
    transitions = make_transitions(td)

    myFSM = FSM(states, transitions)
    chatter_log = myFSM.chatter(240,50)
    save_txt(chatter_log, '/Users/alexten/Projects/PDP/SRN/sandbox/simple-examples/toy_data/')


if __name__=='__main__': main()

