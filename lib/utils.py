import os, csv
import numpy as np
import scipy.spatial
import scipy.io as sio

root_folder = '/home/amax/path/'

def get_atlas_coords(atlas_name='aal'):
    """
    atlas_name : name of the atlas used

    returns:
        matrix : matrix of roi 3D coordinates in MNI space (num_rois x 3)
    """

    coords_file = os.path.join(root_folder, atlas_name + '_coords.csv')
    coords = np.loadtxt(coords_file, delimiter=',')

    if atlas_name == 'ho':
        coords = np.delete(coords, 82, axis=0)

    return coords


def distance_scipy_spatial(z, k=4, metric='euclidean'):
    """Compute exact pairwise distances."""
    d = scipy.spatial.distance.pdist(z, metric)
    d = scipy.spatial.distance.squareform(d)
    # k-NN graph.
    idx = np.argsort(d)[:, 1:k + 1]
    d.sort()
    d = d[:, 1:k + 1]
    return d, idx


def get_ids(num_subjects=None, short=True):
    """
        num_subjects   : number of subject IDs to get
        short          : True of False, specifies whether to get short or long subject IDs

    return:
        subject_IDs    : list of subject IDs (length num_subjects)
    """

    if short:
        subject_IDs = np.loadtxt(os.path.join(root_folder, 'subject_IDs.txt'), dtype=int)
        subject_IDs = subject_IDs.astype(str)
    else:
        subject_IDs = np.loadtxt(os.path.join(root_folder, 'full_IDs.txt'), dtype=str)

    if num_subjects is not None:
        subject_IDs = subject_IDs[:num_subjects]

    return subject_IDs


def load_all_networks(subject_list, kind, atlas_name="aal"):
    """
        subject_list : the subject short IDs list
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        atlas_name   : name of the atlas used

    returns:
        all_networks : list of connectivity matrices (regions x regions)
    """
    all_networks = []

    for subject in subject_list:
        fl = os.path.join(root_folder, subject,subject + "_" + atlas_name + "_" + kind + ".mat")
        #fl = os.path.join('/home/amax/Dataset', subject + "_" + atlas_name + "_" + kind + ".mat")
        matrix = sio.loadmat(fl)['connectivity']

        if atlas_name == 'ho':
            matrix = np.delete(matrix, 82, axis=0)
            matrix = np.delete(matrix, 82, axis=1)

        all_networks.append(matrix)
    # all_networks=np.array(all_networks)

    return all_networks


def get_subject_label(subject_list, label_name):
    """
        subject_list : the subject short IDs list
        label_name   : name of the label to be retrieved

    returns:
        label        : dictionary of subject labels
    """

    label = {}

    with open(os.path.join(root_folder, 'Phenotypic_V1_0b_asd1.csv')) as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            if row['subject'] in subject_list:
                label[row['subject']] = row[label_name]

    return label


def give_idx(number, train_ratio, test_ratio):
    '''
    Split the idx to training ,val, and test mask

        number : the number of samples
        train_ratio : the train set ratio of all setting
        test_ratio : the test set ratio of all setting

        return : the idx of train , test and val
    '''
    number = int(number)
    np.random.seed(42)
    shuffled_indices = np.random.permutation(number)
    train_set_size = int(number * train_ratio)
    test_set_size = int(number * test_ratio)

    train_indices = shuffled_indices[:train_set_size]
    test_indices = shuffled_indices[train_set_size:train_set_size + test_set_size]
    val_indices = shuffled_indices[train_set_size + test_set_size:]

    return train_indices, val_indices, test_indices
