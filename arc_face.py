import numpy as np
from PIL import Image

def arc_similarity(arc, probe, gallery):
    '''
        input images are IOBytes 
    '''


    probe = np.float32(probe)
    gallery = np.float32(gallery)

    emb_probe = arc.calc_emb(probe)
    emb_gallery = arc.calc_emb(gallery)
    return arc.get_distance_embeddings(emb_probe, emb_gallery)

