import scipy


def get_nets_and_ages():
    mat = scipy.io.loadmat("data/nhw2022-network-harmonics-data.mat")
    return mat["nets"], mat["age"].flatten()
