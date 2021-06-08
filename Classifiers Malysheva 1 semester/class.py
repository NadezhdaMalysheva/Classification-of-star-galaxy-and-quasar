import sys
from prediction import OB_GQ, OB_ST, ST
from glob import glob
from os import environ
from os.path import join
import joblib
import numpy as np

def main():
    print('ok')

if __name__ == '__main__':
    if len(sys.argv) < 4:
            print('Usage: %s classifier data_dir name_data proba' % sys.argv[0])
            exit(0)
    mode = sys.argv[1]
    data_dir = sys.argv[2]
    name_data = sys.argv[3]
    if len(sys.argv) > 4:
        proba = True
    else:
        proba = False

    try:
        X = np.load(join(data_dir, name_data))
    except:
        print('Unable to open ', name_data)
        exit(0)

    if mode == 'OB_GQ':
        rez = OB_GQ(data_dir, X, proba)
        #print(rez)
    elif mode == 'OB_ST':
        rez = OB_ST(data_dir, X, proba)
    elif mode == 'ST':
        rez = ST(data_dir, X, proba)
    else:
        print('Usage: %s classifier data_dir name_data proba' % sys.argv[0])
        print('Unknown name_data. Choose: OB_GQ, OB_ST, ST')
        exit(0)

    main()