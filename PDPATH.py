import os

def PDPATH(x=''):
    if x:
        return os.path.dirname(os.path.abspath(__file__)) + x
    else:
        return os.path.dirname(os.path.abspath(__file__))

def main():
    print(PDPATH())
if __name__=='__main__': main()