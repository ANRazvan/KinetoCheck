"""
Constants for the KinetoCheck application.
"""

# Kinect skeleton joint names (25 joints, IntelliRehab dataset)
KINECT_JOINT_NAMES = [
    "SpineBase",        # 0
    "SpineMid",         # 1
    "Neck",             # 2
    "Head",             # 3
    "ShoulderLeft",     # 4
    "ElbowLeft",        # 5
    "WristLeft",        # 6
    "HandLeft",         # 7
    "ShoulderRight",    # 8
    "ElbowRight",       # 9
    "WristRight",       # 10
    "HandRight",        # 11
    "HipLeft",          # 12
    "KneeLeft",         # 13
    "AnkleLeft",        # 14
    "FootLeft",         # 15
    "HipRight",         # 16
    "KneeRight",        # 17
    "AnkleRight",       # 18
    "FootRight",        # 19
    "SpineShoulder",    # 20
    "HandTipLeft",      # 21
    "ThumbLeft",        # 22
    "HandTipRight",     # 23
    "ThumbRight",       # 24
]

# COCO keypoint names (17 joints, YOLO pose estimation)
COCO_KEYPOINT_NAMES = [
    "Nose",             # 0
    "LeftEye",          # 1
    "RightEye",         # 2
    "LeftEar",          # 3
    "RightEar",         # 4
    "LeftShoulder",     # 5
    "RightShoulder",    # 6
    "LeftElbow",        # 7
    "RightElbow",       # 8
    "LeftWrist",        # 9
    "RightWrist",       # 10
    "LeftHip",          # 11
    "RightHip",         # 12
    "LeftKnee",         # 13
    "RightKnee",        # 14
    "LeftAnkle",        # 15
    "RightAnkle",       # 16
]

# IntelliRehab label mapping
# Original CorrectLabel in filename: 1=correct, 2=incorrect, 3=poorly executed
# After label-1: 0=correct, 1=incorrect, 2=poorly executed (skipped)
LABEL_NAMES = {
    0: "correct",
    1: "incorrect",
}
