import os

def PDPATH():
 return os.path.dirname(os.path.abspath(__file__))

def main():
    print(PDPATH())
if __name__=='__main__': main()