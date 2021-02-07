from arcface import ArcFace

face_rec = ArcFace.ArcFace()

def arc_similarity(probe, gallery):
    emb_probe = face_rec.calc_emb(probe)
    emb_gallery = face_rec.calc_emb(gallery)

    return face_rec.get_distance_embeddings(emb_probe, emb_gallery)
