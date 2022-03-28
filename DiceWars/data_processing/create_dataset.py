import numpy as np
import json
import pickle
#from tqdm import tqdm

json_path = 'dataset.json'
save_path = 'dataset.p'
nth = 7

def feed_nn(data):
    g_mat = np.zeros((34,34))
    zzz = [[area['id'], n] for area in data.values() for n in area['neighbours'].values() if n]
    points = np.array([[z[0], zz] for z in zzz for zz in z[1]]) - 1
    for point in points:
        g_mat[point[0], point[1]] = 1
    g_mat += np.eye(34)
    return g_mat
                
def create_dataset_class(every_nth, data_path, save_path='dataset.p'):
    with open(data_path, 'r') as f:
        _data = f.readlines()
        dataset = []
        for d in _data: # tqdm(_data, ncols=100, desc="Loading and parsing data ..."):
            data = json.loads(d)
            states = data['states'][::every_nth] + [data['states'][-1]]
            g_mat = feed_nn(data['states'][0]['board'])
            for state in states:
                areas = [state['board'][str(a)] for a in range(1,35)]
                dice = np.array([area['dice'] for area in areas]) / 8
                owners = np.zeros((len(state['board']), 4))
                for i, owner in enumerate([area['owner'] for area in areas]):
                    owners[i][owner-1] = 1
                result = [(3-data['result'].index(_id))/6 for _id in [1,2,3,4]]
                dataset.append(
                    ((g_mat, owners, dice),result)
                )
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)
        return dataset

ds4 = create_dataset_class(nth, json_path)
