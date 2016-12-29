import numpy as np
import collections

def pprint(a):
    print(round(a,2))


def printprob(h,ev,data):
    print('p({}|{}) = {}'.format(h, ev, round(post_prob(h,ev,data),4)))
    return post_prob(h,ev,data)


def select(ev, w):
    v = list(ev)
    inds = []
    for i,el in enumerate(list(v)):
        if el != '-': inds.append(i*2) if el=='1' else inds.append(i*2+1)
    inds = np.array(inds)
    return w[inds]


def Si(h,ev,data):
    prior = list(data.keys()).count(h) / len(list(data.keys()))
    likelihood = np.prod(select(ev,data[h]))
    return prior * likelihood

def post_prob(h,ev,data):
    numer = Si(h,ev,data)
    denom = sum([Si(x,ev,data) for x in data.keys()])
    return numer / denom

def main():

    features_of = collections.OrderedDict()
    features_of['T'] = np.array([0.95, 0.05, 0.95, 0.05, 0.05, 0.95])
    features_of['U'] = np.array([0.05, 0.95, 0.05, 0.95, 0.95, 0.05])
    features_of['I'] = np.array([0.95, 0.05, 0.95, 0.05, 0.95, 0.05])

    for k, v in features_of.items(): print(k, v)
    print()

    p1 = printprob('T','010',features_of)
    p2 = printprob('U','010',features_of)
    p3 = printprob('I','010',features_of)

    print('\n{} + {} + {} = {}'.format(p1, p2, p3, p1 + p2 + p3))

if __name__=='__main__': main()