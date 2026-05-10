from pathlib import Path


VICON_39_JOINT_NAMES = [
		"LFHD", "RFHD", "LBHD", "RBHD",
		"C7", "T10", "CLAV", "STRN", "RBAK",
		"LSHO", "LUPA", "LELB", "LFRM", "LWRA", "LWRB", "LFIN",
		"RSHO", "RUPA", "RELB", "RFRM", "RWRA", "RWRB", "RFIN",
		"LASI", "RASI", "LPSI", "RPSI",
		"LTHI", "LKNE", "LTIB", "LANK", "LHEE", "LTOE",
		"RTHI", "RKNE", "RTIB", "RANK", "RHEE", "RTOE",
	]

SEQUENCE_LENGTH = 120
UIPRMD_KEYPOINT_DIM = 3

UIPRMD_PATH = Path("Datasets") / "UIPRMD"

UIPRMD_SOURCES_VICON = [
    (UIPRMD_PATH / "Segmented Movements" / "Vicon" / "Positions", 0, "segmented_correct", "*.txt"),
    (UIPRMD_PATH / "Incorrect Segmented Movements" / "Vicon" / "Positions", 1, "segmented_incorrect", "*.txt"),
    (UIPRMD_PATH / "Movements" / "Vicon" / "Positions", 0, "full_correct", "*.txt"),
    (UIPRMD_PATH / "Incorrect Movements" / "Vicon" / "Positions", 1, "full_incorrect", "*.txt"),
]

UIPRMD_SOURCES_KINECT = [
    (UIPRMD_PATH / "Segmented Movements" / "Kinect" / "Positions", 0, "segmented_correct", "*.txt"),
    (UIPRMD_PATH / "Incorrect Segmented Movements" / "Kinect" / "Positions", 1, "segmented_incorrect", "*.txt"),
    (UIPRMD_PATH / "Movements" / "Kinect" / "Positions", 0, "full_correct", "*.txt"),
    (UIPRMD_PATH / "Incorrect Movements" / "Kinect" / "Positions", 1, "full_incorrect", "*.txt"),
]