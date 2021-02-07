import numpy as np
from PIL import Image



def arc_similarity(arc, probe, gallery):
    p_image_bytes = Image.open(probe)
    g_image_bytes = Image.open(gallery)

    probe = np.float32(np.asarray(p_image_bytes))
    gallery = np.float32(np.asarray(g_image_bytes))

    emb_probe = arc.calc_emb(probe)
    emb_gallery = arc.calc_emb(gallery)
    return arc.get_distance_embeddings(emb_probe, emb_gallery)
    
