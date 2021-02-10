import numpy as np
from PIL import Image

def arc_similarity(arc, probe, emb_gallery):
    '''
        input images are IOBytes 
    '''
    
    probe = np.float32(probe)
    emb_probe = arc.calc_emb(probe)
    return arc.get_distance_embeddings(emb_probe, emb_gallery)

