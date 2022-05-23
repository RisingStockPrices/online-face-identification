import face_recognition
import numpy as np
import cv2

class FaceInstance:
    def __init__(self,fno,bbox,embedding,img):
        self.video_pth = None
        self.frame_no = fno
        
        self.face_id = None  # which cluster
        self.bbox = bbox
        self.embedding = embedding
        self.img = img # cropped image of face
    
    def get_embedding(self):
        return self.embedding
    
    def get_img(self):
        return self.img

    def visualize(self,fname=None):
        if fname is None:
            raise NotImplementedError
        
        cv2.imwrite(fname,self.img)

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


class FaceCluster:
    def __init__(self,face):
        self.label = None
        self.cluster_center = face.get_embedding()
        self.faces = [face] # list of FACEINSTANCES
    
    def get_center(self):
        return self.cluster_center
    
    def add_face(self,face):
        # adjust center
        new_center = self.cluster_center*len(self.faces) + face.get_embedding()
        self.faces.append(face)
        self.cluster_center = new_center / len(self.faces)

    def visualize(self,fname=None,no=10):
        if fname is None:
            raise NotImplementedError
        
        size = (50,50)
        # retrieve top 5 (tmp)
        no = min(len(self.faces),no)
        imgs = [cv2.resize(f.get_img(),size) for f in self.faces[:no]]
        img_cluster = cv2.hconcat(imgs)

        cv2.imwrite(fname,img_cluster)


class FaceClusterLibrary:
    """System for semantic face ids"""
    def __init__(self):
        self.clusters = dict() # group by cluster id
        self.frames = dict() # group by frame (dealing with single video for now)
        self.threshold = None
    
    def find_closest_cluster(self,embedding,threshold=None):
        """Iterate thru all clusters to get closest cluster center"""
        encodings = []
        encodings_keys = []
        
        if len(self.clusters)==0:
            return None

        for k,cluster in self.clusters.items():
            encodings.append(cluster.get_center())
            encodings_keys.append(k)
        
        face_distances = face_recognition.face_distance(encodings,embedding)
        best_idx = np.argmin(face_distances)
        best_match = encodings_keys[best_idx]
        best_dist = face_distances[best_idx]
        
        if best_dist > threshold:
            return None
        
        print(face_distances)
        return best_match
    
    def add_face_cluster(self,face):
        print("creating new face in system")
        new_cluster = FaceCluster(face)
        new_idx = len(self.clusters)
        self.clusters[new_idx] = new_cluster
        # TODO: add to per-frame dict

        return new_idx

    def add_face_instance(self,face,idx):
        print("adding face to pre-existing cluster %d" % idx)
        self.clusters[idx].add_face(face)
        # TODO: add to per-frame dict
        import pdb;pdb.set_trace()
        self.clusters[0].visualize('test.png')

    def process_faces(self,faces):
        for face in faces:
            # import pdb;pdb.set_trace()
            res = self.find_closest_cluster(face.get_embedding())
            if res is None:
                # create new cluster
                self.add_face_cluster(face)
            else:
                self.add_face_instance(face,res)
            
            

cluster_library = FaceClusterLibrary()