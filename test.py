import face_recognition
import cv2
import time
from time import strftime, gmtime

from cluster import *

TEST_VIDEO = '/vsa/face-clustering/videos/test.mp4'
OUT_VIDEO = '/vsa/face-clustering/results/output.avi'

def get_elapsed_time(start_time,end_time=None):
    """Returns elapsed time from start in fixed string format"""
    if end_time is None:
        end_time = time.time()
    
    return strftime('%M:%S',gmtime(end_time-start_time))

def draw_bounding_box(frame, bbox, label=None,color=None):
    top,right,bottom,left = bbox

    if color is None: # default color
        color = (0,0,255)
    # import pdb;pdb.set_trace()
    cv2.rectangle(frame,(left,top),(right,bottom),color,2)
    if label is not None:
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, label, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    
def test_detection(input_vid,output_vid,max_frames=None):
    """Find all faces in video and output them"""
    vid = cv2.VideoCapture(input_vid)
    if max_frames is None:
        max_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_size = (int(vid.get(3)),int(vid.get(4)))
    out = cv2.VideoWriter(OUT_VIDEO,cv2.VideoWriter_fourcc(*'XVID'),29.97,frame_size)

    frame_count = 0
    print("Start face detection for video %s" % output_vid)
    start_time = time.time()
    while True:
        ret, frame = vid.read()
        frame_count += 1
        if not ret:
            break
        # face detection
        _frame = frame[:, :, ::-1] # color scheme conversion (cv2 <-> face_recognition)
        # TODO: batch processing to make use of gpus
        face_locations = face_recognition.face_locations(_frame)
        
        for bbox in face_locations:
            # Draw bounding box - (top,right,bottom,left) ordering
            draw_bounding_box(frame,bbox)
        
        out.write(frame)
        
        if frame_count >= max_frames:
            break
        elif frame_count % 100 == 0:
            print("[%s] Done processing %d frames" % (get_elapsed_time(start_time),frame_count))
            
    print("[%s] Done. Total %d frames processed" % (get_elapsed_time(start_time),frame_count))
    out.release()

def test_embedding(input_vid,output_vid,max_frames=None):
    """test face encoding module and cluster system"""
    vid = cv2.VideoCapture(input_vid)
    if max_frames is None:
        max_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_size = (int(vid.get(3)),int(vid.get(4)))
    out = cv2.VideoWriter(OUT_VIDEO,cv2.VideoWriter_fourcc(*'XVID'),29.97,frame_size)

    frame_count = 0
    start_time = time.time()
    while True:
        ret, frame = vid.read()
        frame_count += 1
        if not ret:
            break
        # face detection
        _frame = frame[:, :, ::-1] # color scheme conversion (cv2 <-> face_recognition)
        # TODO: batch processing to make use of gpus
        face_locations = face_recognition.face_locations(_frame)
        
        if len(face_locations) > 1:
            import pdb;pdb.set_trace()
            face_instances = extract_face_instances(frame,frame_count,face_locations)
            cluster_library.process_faces(face_instances)

        # for bbox in face_locations:
        #     cluster_library.process_faces
        #     # Draw bounding box - (top,right,bottom,left) ordering
        #     draw_bounding_box(frame,bbox)
        
        # out.write(frame)
        
        if frame_count >= max_frames:
            break
        elif frame_count % 100 == 0:
            print("[%s] Done processing %d frames" % (get_elapsed_time(start_time),frame_count))
            
    print("[%s] Done. Total %d frames processed" % (get_elapsed_time(start_time),frame_count))
    out.release()


if __name__=="__main__":
    test_embedding(TEST_VIDEO,'embedding.avi',200)
