import time, threading
import argparse
import time
import cv2
import numpy as np
import math
import os
import mediapipe as mp
from numpy import interp
import uuid
from typing import Mapping, Tuple
from mediapipe.python.solutions import drawing_styles
import pygame
from OpenGL.GL import *

MIN_MATCHES = 15
DEFAULT_COLOR = (0, 255, 0)
dir_name = os.getcwd()
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

def findArucoMarkers(image, markerSize=6, totalMarkers=250):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dictionary_key = getattr(cv2.aruco, f'DICT_{markerSize}X'
                                        f'{markerSize}_{totalMarkers}')
    aruco_dictionary = cv2.aruco.getPredefinedDictionary(dictionary_key)
    aruco_params = cv2.aruco.DetectorParameters()
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dictionary,
                                                            parameters=aruco_params)
    return marker_corners, marker_ids


def superimposeImageOnMarkers(video_frame, aruco_markers, obj , overlay_image,
                              video_width, video_height):
    frame_height, frame_width = video_frame.shape[:2]
    # print(aruco_markers[0][0][0][0])
    if len(aruco_markers[0]) != 0:
        for i, marker_corner in enumerate(aruco_markers[0]):
            marker_corners = marker_corner.reshape((4, 2)).astype(np.int32)
            cv2.polylines(video_frame, [marker_corners], True, (0, 255, 0), 2)
            cv2.putText(video_frame, str(aruco_markers[1][i]),
                        tuple(marker_corners[0]),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)
            homography_matrix, _ = cv2.findHomography(
                np.array([[0, 0], [video_width, 0], [video_width, video_height],
                          [0, video_height]], dtype="float32"),marker_corners)
            warped_image = cv2.warpPerspective(overlay_image, homography_matrix,
                                               (frame_width, frame_height))
            mask = np.zeros((frame_height, frame_width), dtype="uint8")
            cv2.fillConvexPoly(mask, marker_corners, (255, 255, 255), cv2.LINE_AA)

            masked_warped_image = cv2.bitwise_and(warped_image, warped_image,
                                                  mask=mask)
            masked_video_frame = cv2.bitwise_and(video_frame, video_frame,
                                                 mask=cv2.bitwise_not(mask))
            # h = aruco_markers[0][0][0][0][1]
            video_frame = cv2.add((augment(frame, obj, projection = projection, h = 50, w = 50, scale=50)), 0)
            # video_frame = cv2.add(masked_warped_image, masked_video_frame, (augment(frame, obj, projection = projection, h = 50, w = 50, scale=100)))
            # video_frame = renderObj(video_frame, obj, projection= projection, color=True)
    return video_frame

def augment(img, obj, projection, h, w, scale = 4):
    h, w = h, w
    vertices = obj.vertices
    img = np.ascontiguousarray(img, dtype=np.uint8)

    #projecting the faces to pixel coords and then drawing
    for face in obj.faces:
        #a face is a list [face_vertices, face_tex_coords, face_col]
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices]) #-1 because of the shifted numbering
        points = scale*points
        points = np.array([[p[2] + w/2, p[0] + h/2, p[1]] for p in points]) #shifted to centre 
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)#transforming to pixel coords
        imgpts = np.int32(dst)
        cv2.fillConvexPoly(img, imgpts, face[-1])
        
    return img

def MTL(filename):
    contents = {}
    mtl = None
    for line in open(filename, "r"):
        if line.startswith('#'): continue
        values = line.split()
        if not values: continue
        if values[0] == 'newmtl':
            mtl = contents[values[1]] = {}
        elif mtl is None:
            raise ValueError("mtl file doesn't start with newmtl stmt")
        elif values[0] == 'map_Kd':
            mtl[values[0]] = values[1]
            surf = pygame.image.load("/".join(list(filename.split('/')[0:-1]))+"/"+mtl['map_Kd'])
            image = pygame.image.tostring(surf, 'RGBA', 1)
            ix, iy = surf.get_rect().size
            texid = mtl['texture_Kd'] = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texid)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA,
                GL_UNSIGNED_BYTE, image)
        else:
            mtl[values[0]] = map(float, values[1:])
    return contents

class OBJ:
    def __init__(self, filename, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        material = None
        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(map(float, values[1:3]))
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, norms, texcoords))

def render(img, obj, projection, model, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 50
    h, w = model.shape
    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, DEFAULT_COLOR)
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1] 
            cv2.fillConvexPoly(img, imgpts, color)

    return img

def renderObj(img, obj, projection, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 50
    h, w = (644,372)
    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, DEFAULT_COLOR)
        else:
            color = face[-1]
            color = color[::-1]
            cv2.fillConvexPoly(img, imgpts, color)        

    return img

def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    print(hex_color)
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))

def init_feature(name):
    chunks = name.split('-')
    if chunks[0] == 'sift':
        detector = cv2.xfeatures2d.SIFT_create()
        norm = cv2.NORM_L2
    elif chunks[0] == 'surf':
        detector = cv2.xfeatures2d.SURF_create(800)
        norm = cv2.NORM_L2
    elif chunks[0] == 'orb':
        detector = cv2.ORB_create(200)
        norm = cv2.NORM_HAMMING
    elif chunks[0] == 'akaze':
        detector = cv2.AKAZE_create()
        norm = cv2.NORM_HAMMING
    elif chunks[0] == 'brisk':
        detector = cv2.BRISK_create()
        norm = cv2.NORM_HAMMING
    else:
        return None, None
    if 'flann' in chunks:
        if norm == cv2.NORM_L2:
            flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        else:
            flann_params= dict(algorithm = FLANN_INDEX_LSH,
                               table_number = 6,
                               key_size = 12,     
                               multi_probe_level = 1) 
        matcher = cv2.FlannBasedMatcher(flann_params, {}) 
    else:
        matcher = cv2.BFMatcher(norm)
    return detector, matcher


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def getColor(zDist):
    c = int(interp(zDist, [0,15], [0,255]))
    return (c,c,c)

def createLandMarks(hand_landmarks):
  hand_landmark_style = {}  
  for k, v in drawing_styles._HAND_LANDMARK_STYLE.items():
    for landmark in k:
      c = getColor(abs(hand_landmarks.landmark[landmark].z*100))
      r = int(abs(hand_landmarks.landmark[landmark].z*100))
      hand_landmark_style[landmark] =   mp_drawing.DrawingSpec(color=c, thickness=drawing_styles._THICKNESS_DOT, circle_radius= r )
  return hand_landmark_style       

def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    print("What")
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)

cap = cv2.VideoCapture(0)
# Load 3D model from OBJ file 
camera_parameters = np.array([[1.01937196e+03, 0.00000000e+00, 6.18709801e+02],
 [0.00000000e+00, 1.02421390e+03, 3.27280523e+02], [0, 0, 1]] )

homography =  np.float32([[0.4160569997384721, -1.306889006892538, 553.7055461075881],
                          [0.7917584252773352, -0.06341244158456338, -108.2770029401219],
                          [0.0005926357240956578, -0.001020651672127799, 1]])
projection = projection_matrix(camera_parameters, homography)
createControls = 1
counter = 0

def on_change(value):
    valuelf = value/360
    print(valuelf)
    homography[1][1] = valuelf    

choice = 1

def run(export_path):
    obj1 = OBJ(export_path, swapyz=True)  
    obj2 = OBJ(export_path, swapyz=True) 
    if (choice == 1):    
        with mp_hands.Hands(static_image_mode=False,min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
            while cap.isOpened():
                ret, frame = cap.read()
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = cv2.flip(image, 1)
                image.flags.writeable = False
                results = hands.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image_height, image_width, _ = image.shape
                
                # Rendering results
                if results.multi_hand_landmarks:
                    for num, hand_landmarks  in enumerate(results.multi_hand_landmarks):
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                                            createLandMarks(hand_landmarks),
                                            mp_drawing_styles.get_default_hand_connections_style())                        
                                        
                        lnd1 = hand_landmarks.landmark[4]
                        lnd2 = hand_landmarks.landmark[0]
                        lnd3 = hand_landmarks.landmark[17]
                        lnd4 = hand_landmarks.landmark[8]
                        lndLst = np.array([[lnd1.x*image_width, lnd1.y* image_height],
                                        [lnd2.x*image_width, lnd2.y* image_height],
                                        [lnd3.x*image_width, lnd3.y* image_height], 
                                        [lnd4.x*image_width, lnd4.y* image_height],
                                        [lnd1.x*image_width, lnd1.y* image_height]]).reshape((-1, 1, 2))
                        
                        image = cv2.polylines(image, [np.int32(lndLst)], True, 255, 3, cv2.LINE_AA)
                        
                        src_pts = np.float32([0 , 0 ,
                                            500, 0,
                                            500, 500,
                                            0, 500]).reshape(-1, 1, 2)
                        dst_pts = np.float32([lnd1.x*image_width, lnd1.y* image_height,
                                            lnd2.x*image_width, lnd2.y* image_height,
                                            lnd3.x*image_width, lnd3.y* image_height,
                                            lnd4.x*image_width, lnd4.y* image_height]).reshape(-1, 1, 2) 
                        dst_pts = dst_pts.round(2)
                        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        projection = projection_matrix(camera_parameters, homography)  
                        
                        if(results.multi_handedness[num].classification[0].label == "Left"):
                            image = renderObj(image, obj1, projection, True)
                        else:
                            image = renderObj(image, obj2, projection, True)
                
                plot = np.zeros([image_height, image_width, 3], dtype=np.uint8)                
                if results.multi_hand_world_landmarks:
                    for num,hand_world_landmarks in enumerate(results.multi_hand_world_landmarks):                
                        for idx,landMrk in enumerate(hand_world_landmarks.landmark):
                            hand_world_landmarks.landmark[idx].x += 0.5
                            hand_world_landmarks.landmark[idx].y += 0.5
                        mp_drawing.draw_landmarks(plot,hand_world_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # cv2.imshow('Plot', plot)
                cv2.imshow('HandTracking', image) 

                if(createControls):
                    createControls = 0
                
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
    else:
        video_height = 600
        video_width = 800
        overlay_image = cv2.imread('overlay.jpg')
        overlay_image = cv2.resize(overlay_image, (video_width, video_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                aruco_markers = findArucoMarkers(frame, totalMarkers=1000)
                frame = superimposeImageOnMarkers(frame, aruco_markers, obj1, overlay_image,
                                                    video_width, video_height)
                cv2.imshow('Video', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break


cap.release()
cv2.destroyAllWindows()
