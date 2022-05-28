from .cluster import * 

def extract_face_instances(frame,frame_idx,bboxes=None):
    """ Extracts list of face instances from single frame """
    if bboxes is None:
        # TODO: do detection also
        raise NotImplemnetedError

    face_instances = []
    face_encodings = face_recognition.face_encodings(frame,bboxes)
    for bbox,encoding in zip(bboxes,face_encodings):
        img = frame[bbox[0]:bbox[2],bbox[3]:bbox[1]]
        face_instance = FaceInstance(frame_idx,bbox,encoding,img)
        face_instances.append(face_instance)
    
    return face_instances