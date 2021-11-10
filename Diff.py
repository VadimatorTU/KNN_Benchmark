import numpy as np
AMOUNT = 1000000

for attempt in range(1, 11, 1):
    etalon = np.load(f'Output/Brute_force/Brute_force_{AMOUNT}_{attempt}.npy')
    for leaf_size in range(1, 11, 3):
        to_proof = np.load(f'Output/KDTree/KDTree_{AMOUNT}_{leaf_size}_{attempt}.npy')
        difference = np.setdiff1d(to_proof, etalon)
        print(f'Attempt {attempt}, {leaf_size} leaf size')
        print(difference.size)
    print('===========================================================')


for attempt in range(1, 11, 1):
    etalon = np.load(f'Output/Brute_force/Brute_force_{AMOUNT}_{attempt}.npy')

    for np_tables in range(2, 5, 1):
        to_proof = np.load(f'Output/LSH/LSH_{AMOUNT}_2_{np_tables}_{attempt}.npy')
        difference = np.setdiff1d(to_proof, etalon)
        print(f'Attempt {attempt} 2 projections {np_tables} tables')
        print(difference.size)
    print('===========================================================')

    for np_tables in range(2, 5, 1):
        to_proof = np.load(f'Output/LSH/LSH_{AMOUNT}_5_{np_tables}_{attempt}.npy')
        difference = np.setdiff1d(to_proof, etalon)
        print(f'Attempt {attempt} 5 projections {np_tables} tables')
        print(difference.size)
    print('===========================================================')

    for np_tables in range(2, 5, 1):
        to_proof = np.load(f'Output/LSH/LSH_{AMOUNT}_10_{np_tables}_{attempt}.npy')
        difference = np.setdiff1d(to_proof, etalon)
        print(f'Attempt {attempt} 10 projections {np_tables} tables')
        print(difference.size)
    print('===========================================================')