import face_recognition
import cv2

from utils.visualize import *
instance_count = 0

class FaceInstance:
    def __init__(self,fno,bbox,embedding,img):
        self.video_pth = None
        self.frame_no = fno
        
        global instance_count
        self.instance_id = instance_count + 1
        instance_count += 1

        self.face_id = None  # which cluster
        self.bbox = bbox
        self.embedding = embedding
        self.img = img # cropped image of face
    
    def set_face_id(self,id):
        self.face_id = id

    def get_face_id(self):
        return self.face_id
    
    def get_bbox(self):
        return self.bbox

    def get_embedding(self):
        return self.embedding
    
    def get_fno(self):
        return self.frame_no
    
    def get_img(self):
        return self.img

    def visualize(self,fname=None,write=True):
        if fname is None and write is True:
            raise NotImplementedError
        
        size = (50,50)
        img = cv2.resize(self.img,size)
        img = cv2.vconcat([img,np.zeros((20,50,3),np.uint8)])
        # put text
        cv2.putText(img,str(self.instance_id), (5,65), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255),1)
        if write:
            cv2.imwrite(fname,img)
        return img

    
    def __str__(self):
        return 'Instance ID: %d\nFrame number %d' % (self.instance_id,self.frame_no)


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

    def visualize(self,fname=None,no=10,write=True):
        if fname is None and write:
            raise NotImplementedError
        
        size = (50,50)
        # retrieve top 5 (tmp)
        no_ = min(len(self.faces),no)
        imgs = [f.visualize(write=False) for f in self.faces[:no_]]
        for i in range(no-len(imgs)):
            imgs.append(np.zeros((70,50,3),np.uint8))
        img_cluster = cv2.hconcat(imgs)

        if write:
            cv2.imwrite(fname,img_cluster)
        return img_cluster
    
    def __str__(self):
        return ''


class FaceClusterLibrary:
    """System for semantic face ids"""
    def __init__(self):
        self.clusters = dict() # group by cluster id
        self.frames = dict() # group by frame (each frame holds a list of faces)
        self.threshold = 0.4
    
    def find_closest_cluster(self,embedding):
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
        
        if best_dist > self.threshold:
            return None
        
        print(face_distances)
        return best_match
    
    def get_faces_in_frame(self,fno):
        faces = self.frames[fno]
        ids = [face.get_face_id() for face in faces]
        return ids

    def add_face_to_frame_list(self,face):
        fno = face.get_fno()
        self.frames[fno].append(face)

    def add_face_cluster(self,face):
        print("creating new face in system")
        new_idx = len(self.clusters)
        face.set_face_id(new_idx)
        new_cluster = FaceCluster(face)
        self.clusters[new_idx] = new_cluster
        # TODO: add to per-frame dict
        return new_idx

    def add_face_instance(self,face,idx):
        print("adding face to pre-existing cluster %d" % idx)
        face.set_face_id(idx)
        self.clusters[idx].add_face(face)
        # TODO: add to per-frame dict

    def process_faces(self,faces):
        self.frames[faces[0].frame_no] = []

        for face in faces:
            # import pdb;pdb.set_trace()
            res = self.find_closest_cluster(face.get_embedding())
            if res is None:
                # create new cluster
                self.add_face_cluster(face)
            else:
                self.add_face_instance(face,res)
            
            self.add_face_to_frame_list(face)
    
    def visualize_frame(self,frame,fno,write=False):
        if fno in self.frames.keys():
            faces = self.frames[fno]
        else:
            faces = []
        for face in faces:
            face_id = face.get_face_id()
            bbox = face.get_bbox()
            draw_bounding_box(frame,bbox=bbox,label=str(face_id),color=COLOR[face_id])
        
        if write:
            # import pdb;pdb.set_trace()
            cv2.imwrite('debug.png',frame)
        return frame

    def visualize(self,fname=None,no=5,write=True):
        if fname is None and write:
            raise NotImplementedError
        
        no_cluster = 10
        no_ = min(len(self.clusters),no)
        clusters = []
        for idx,(key,cluster) in enumerate(self.clusters.items()):
            clusters.append(cluster.visualize(no=no_cluster,write=False))
            if idx >= no_:
                break
        # clusters = [f.visualize(no=no_cluster,write=False) for f in list(self.clusters.items())[:no]]
        # for i in range(no-len(clusters)):
        #     clusters.append(np.zeros((70,no_cluster*50,3),np.uint8))
        img = cv2.vconcat(clusters)

        if write:
            cv2.imwrite(fname,img)
        return img


cluster_library = FaceClusterLibrary()