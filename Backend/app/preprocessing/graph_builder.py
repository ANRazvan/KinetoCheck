"""
Skeleton graph definitions for different formats.
Used by the ST-GAT model to build spatial attention graphs.
"""


# Kinect skeleton adjacency (25 joints) — IntelliRehab dataset
# Joint indices:
#   0=SpineBase, 1=SpineMid, 2=Neck, 3=Head,
#   4=ShoulderLeft, 5=ElbowLeft, 6=WristLeft, 7=HandLeft,
#   8=ShoulderRight, 9=ElbowRight, 10=WristRight, 11=HandRight,
#   12=HipLeft, 13=KneeLeft, 14=AnkleLeft, 15=FootLeft,
#   16=HipRight, 17=KneeRight, 18=AnkleRight, 19=FootRight,
#   20=SpineShoulder, 21=HandTipLeft, 22=ThumbLeft,
#   23=HandTipRight, 24=ThumbRight
KINECT_EDGES = [
    (0, 1), (1, 20), (20, 2), (2, 3),                        # spine + head
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),      # left arm + hand
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),  # right arm + hand
    (0, 12), (12, 13), (13, 14), (14, 15),                    # left leg
    (0, 16), (16, 17), (17, 18), (18, 19),                    # right leg
]

KINECT_NUM_JOINTS = 25

# COCO skeleton adjacency (17 keypoints) — YOLO pose estimation
# Joint indices:
#   0=Nose, 1=LeftEye, 2=RightEye, 3=LeftEar, 4=RightEar,
#   5=LeftShoulder, 6=RightShoulder, 7=LeftElbow, 8=RightElbow,
#   9=LeftWrist, 10=RightWrist, 11=LeftHip, 12=RightHip,
#   13=LeftKnee, 14=RightKnee, 15=LeftAnkle, 16=RightAnkle
COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12),                # torso
    (11, 13), (13, 15), (12, 14), (14, 16),    # legs
]

COCO_NUM_JOINTS = 17
