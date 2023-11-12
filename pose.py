import cv2
import sys
import math
import torch
import threading
import numpy as np
import tkinter as tk
import mediapipe as mp
import matplotlib.pyplot as plt
from mediapipe import solutions
from tkinter.simpledialog import askstring
from tkinter.filedialog import askopenfilename
from mediapipe.framework.formats import landmark_pb2
from segment_anything import sam_model_registry, SamPredictor
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

body_coord = [
  "NOSE",
  "RIGHT_EYE",
  "LEFT_EYE",
  "RIGHT_EAR",
  "LEFT_EAR",
  "RIGHT_SHOULDER",
  "LEFT_SHOULDER",
  "RIGHT_ELBOW",
  "LEFT_ELBOW",
  "RIGHT_WRIST",
  "LEFT_WRIST",
  "RIGHT_HIP",
  "LEFT_HIP", 
  "RIGHT_KNEE",
  "LEFT_KNEE",
  "RIGHT_ANKLE",
  "LEFT_ANKLE",
  "RIGHT_HEEL",
  "LEFT_HEEL",
]

res = []
# p = "./"

def get_body_coord(str, image_width, image_height, results):
  return np.array([results.pose_landmarks.landmark[mp_holistic.PoseLandmark[str]].x * image_width,
          results.pose_landmarks.landmark[mp_holistic.PoseLandmark[str]].y * image_height])

def input_from_bro(p):
  print("inputting")
  with mp_pose.Pose(
    static_image_mode=True, min_detection_confidence=0.5) as pose:
    print(p)
    image = cv2.imread(p)
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    print(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height})'
    )

    for e in body_coord:
      res.append(get_body_coord(e, image_width, image_height, results))

    # Draw pose landmarks on the image.
    annotated_image = image.copy()
    # Use mp_pose.UPPER_BODY_POSE_CONNECTIONS for drawing below when
    # upper_body_only is set to True.
    mp_drawing.draw_landmarks(
        annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imwrite('./clothes.png', annotated_image)

## model implementation
print(type(res))

def distance_calculator(arr, coord):
  # arr: two dimension array contain tuples (start, end)
  res = []
  for p1, p2 in arr:
    diff = coord[p1] - coord[p2]

    distance = np.sqrt(np.sum(np.square(diff)))
    res.append(distance)
  return res

# Calculating angle opposite to egde b
def angle(a,b,c):
   '''
   @param:
   - a: edge that opposite to the angle that we looking for
   - b, c: edges that beside the angle that we looking for
   '''
   angle_rad = math.acos((b**2 + c**2 - a**2) / (2 * b * c))
   return angle_rad
   # Comment out when needed
#    angle_deg = math.degrees(angle_rad)
#    return angle_deg

def make_angles(coords):
  '''
  A function that generate angles between joints
  @param:
  @return:
  an array that contains the following angles
  - left-shoulder
  - left-elbow
  - right-shoulder
  - right-elbow
  - left-hip
  - right-hip
  '''
  # get first person
  person = coords

  # get points for angle calculation
  points = [[(6, 7), (5, 7), (5, 6)]      ,
            [(5, 8), (5, 6), (6, 8)]      ,
            [(5, 9), (5, 7), (7, 9)]      ,
            [(6, 10), (6, 8), (8, 10)]    ,
            [(12, 13), (11, 12), (11, 13)],
            [(11, 14), (11, 12), (12, 14)]]
  keys = ["left_shoulder" ,
          "right_shoulder",
          "left_elbow"    ,
          "right_elbow"   ,
          "left_hip"      ,
          "right_hip"]
  # angle calculation
  res = {}
  for i, e in enumerate(points):
    ans = angle(*distance_calculator(e, person))
    res[keys[i]] = ans

  return res


# Calculating length of joints of the person (dictionary version)
'''
- pred_coords variable contains joints of many people.
- It has the dimension of 3d where
  + The first dimension denote number of people appear in that image
  + The second dimension denote all of the joints that person has
  + The third dimension denote each joint of that person
'''

'''
@Params:
- coord: all informations about people(s) joints
@Returns:
- dictionary containing desirable data
'''
def joints_length(coords):
  # access to the first person in the image
  person = coords

  '''
  0. left shoulder -> left elbow
  1. left elbow -> left wrist
  2. right shoulder -> right elbow
  3. right elbow -> right wrist
  4. left shoulder -> right shoulder
  5. left shoulder -> left hip
  6. right shoulder -> right hip
  7. left hip -> right hip
  8. left hip -> left knee
  9. right hip -> right knee
  10. left knee -> left ankle
  11. right knee -> right ankle
  '''
  points = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (12, 14), (13, 15), (14, 16)]
  dist = distance_calculator(points, person)

  # debugging purpose
  distance_ref = [
   "left_shoulder-left_elbow",     #0
   "left_elbow-left_wrist",        #1
   "right_shoulder-right_elbow",   #2
   "right_elbow-right_wrist",      #3
   "left_shoulder-right_shoulder", #4
   "left_shoulder-left_hip",       #5
   "right_shoulder-right_hip",     #6
   "left_hip-right_hip",           #7
   "left_hip-left_knee",           #8
   "right_hip-right_knee",         #9
   "left_knee-left_ankle",         #10
   "right_knee-right_ankle"]       #11
  res = {}
  for i, e in enumerate(dist):
    res[distance_ref[i]] = e

  return res

def model(coords):
    y_low = min(coords[17][1], coords[18][1])
    y_high = coords[0][1]
    height = abs(y_high - y_low) * 108.5 / 100

    return {
        "height": lambda: height,
        "lengths": lambda: joints_length(coords),
        "angles": lambda: make_angles(coords)
    }

## Crop image
sys.path.append("..")

# load model
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

if (torch.cuda.is_available()):
    device = "cuda"
else:
    device = "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))


def body_detect(**args):
    
    cols = args["image"].shape[0]
    rows = args["image"].shape[1]

    input_point = np.array([[rows / 2, cols * 0.4], [rows / 2, cols / 2], [rows / 2, cols * 0.65]])
    input_label = np.array([1, 1, 1])

    mask_input = args["logits"][np.argmax(args["scores"]), :, :]  # Choose the model's best mask

    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        mask_input=mask_input[None, :, :],
        multimask_output=False,
    )

    masks.shape

    plt.figure(figsize=(10,10))
    plt.imshow(args["image"])
    show_mask(masks, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.axis('off')
    plt.show()

    for x in range(0, cols - 1):
        for y in range(0, rows - 1):
            if args["mask"][x, y] == False:
                args["image"][x, y] = (127, 255, 0)

    plt.imshow(args["image"])
    plt.axis('off')
    fname = f'./clothes/{args["name"]}_{"front" if args["j"] == 0 else "back"}_body.jpg'
    print(fname)
    plt.savefig(fname)
    
def detect(input_point, input_label, **args):
    print("detecting...")
    image = cv2.imread(args["uploaded"])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask_input = args["logits"][np.argmax(args["scores"]), :, :]  # Choose the model's best mask

    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        mask_input=mask_input[None, :, :],
        multimask_output=False,
    )

    masks.shape

    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(masks, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.axis('off')
    plt.show()

    cols = image.shape[0]
    rows = image.shape[1]
    mask = masks[0]

    for x in range(0, cols - 1):
        for y in range(0, rows - 1):
            if mask[x, y] == False:
                image[x, y] = (127, 255, 0)

    plt.imshow(image)
    plt.axis('off')
    fname = f'./clothes/{args["name"]}_{"front" if args["j"] == 0 else "back"}_{"shirt" if len(input_point) == 4 else "jeans"}.jpg'
    print(fname)
    plt.savefig(fname)
    
# MODEL
def model_SAM():
    print("model_sam")
    global p
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Use a simple dialog to get input instead of input()
    name = askstring("Input", "Please enter something:")

    for j in range(0, 2):
        uploaded = askopenfilename()
        if j == 0:
          p = "./Input_IMG/" + name + ".jpg"
          cv2.imwrite(p, cv2.imread(uploaded))
          print(p)
        image = cv2.imread(uploaded)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        plt.axis('on')
        plt.show()

        predictor.set_image(image)

        height = image.shape[0]
        width = image.shape[1]
        input_point = np.array([[width / 2, height / 2]])
        input_label = np.array([1])

        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_points(input_point, input_label, plt.gca())
        plt.axis('on')
        plt.show()
        
        # From here!
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        masks.shape  # (number_of_masks) x H x W

        

        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            show_mask(mask, plt.gca())
            show_points(input_point, input_label, plt.gca())
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show()

        # Parameters

        dic = {
            "j": j,
            "uploaded": uploaded,
            "image": image,
            "masks": masks,
            "mask": mask,
            "scores": scores,
            "logits": logits,
            "name": name
        }


        """BODY DETECT"""
        
        body_detect(**dic)

        """SHIRT DETECT"""

        height = image.shape[0]
        width = image.shape[1]
        input_point = np.array([[width / 2, height * 0.4], [width / 2, height / 2], [width * 0.45 , height * 0.75], [width * 0.45, height * 0.65]])
        input_label = np.array([1, 1, 0, 0])
        detect(input_point, input_label, **dic)

        """JEANS DETECT"""

        height = image.shape[0]
        width = image.shape[1]
        input_point = np.array([[width * 0.45, height * 0.65], [width * 0.45, height * 0.7], [width * 0.55, height * 0.65], [width * 0.55, height * 0.7], [width / 2, height / 2], [width / 2, height * 0.4]])
        input_label = np.array([1, 1, 1, 1, 0, 0])
        detect(input_point, input_label, **dic)

        return True
        

# model_SAM()
# input_from_bro(p)        
# res = model(res)

# # Sample for retrieving elements:
# res["height"]()
# print(res["lengths"]())
# print(res["angles"]())








