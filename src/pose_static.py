import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def printCoordinates(landmarks):
    noseIndex = mp_pose.PoseLandmark.NOSE
    print(f'Nose: {noseIndex} {landmarks[noseIndex].x}, {landmarks[noseIndex].y}, {landmarks[noseIndex].z}')
    leftHipIndex = mp_pose.PoseLandmark.LEFT_HIP
    print(
        f'LeftHip: {leftHipIndex} {landmarks[leftHipIndex].x}, {landmarks[leftHipIndex].y}, {landmarks[leftHipIndex].z}')
    rightHipIndex = mp_pose.PoseLandmark.RIGHT_HIP
    print(
        f'RightHip: {rightHipIndex} {landmarks[rightHipIndex].x}, {landmarks[rightHipIndex].y}, {landmarks[rightHipIndex].z}')
    leftWristIndex = mp_pose.PoseLandmark.LEFT_WRIST
    print(
        f'LeftWrist: {leftWristIndex} {landmarks[leftWristIndex].x}, {landmarks[leftWristIndex].y}, {landmarks[leftWristIndex].z}')
    rightWristIndex = mp_pose.PoseLandmark.RIGHT_WRIST
    print(
        f'RightWrist: {rightWristIndex} {landmarks[rightWristIndex].x}, {landmarks[rightWristIndex].y}, {landmarks[rightWristIndex].z}')
    leftAnkleIndex = mp_pose.PoseLandmark.LEFT_ANKLE
    print(
        f'LeftAnkle: {leftAnkleIndex} {landmarks[leftAnkleIndex].x}, {landmarks[leftAnkleIndex].y}, {landmarks[leftAnkleIndex].z}')
    rightAnkleIndex = mp_pose.PoseLandmark.RIGHT_ANKLE
    print(
        f'RightAnkle: {rightAnkleIndex} {landmarks[rightAnkleIndex].x}, {landmarks[rightAnkleIndex].y}, {landmarks[rightAnkleIndex].z}\n')


def pose(file_names: [str]):
    # For static images:

    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5) as pose:
        for idx, file in enumerate(file_names):
            print(f'File: {file}')
            image = cv2.imread(file)
            image_height, image_width, _ = image.shape
            # Convert the BGR image to RGB before processing.
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if not results.pose_landmarks:
                print('No results')
                continue

            printCoordinates(results.pose_landmarks.landmark)

            # save_anottated_image(idx, image, results)

            # Plot pose world landmarks.
            mp_drawing.plot_landmarks(
                results.pose_landmarks, mp_pose.POSE_CONNECTIONS, elevation=0, azimuth=90)


def save_anottated_image(idx, image, results):
    BG_COLOR = (192, 192, 192)  # gray
    annotated_image = image.copy()
    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    annotated_image = np.where(condition, annotated_image, bg_image)
    # Draw pose landmarks on the image.
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
    cv2.imshow('MediaPipe Pose', cv2.flip(annotated_image, 1))


if __name__ == '__main__':
    arms_bottom_files = [
        '/home/gilcu2/prog/unity/paint_balls/Assets/Media/armsBottom1.png',
        '/home/gilcu2/prog/unity/paint_balls/Assets/Media/armsBottom2.png',
        '/home/gilcu2/prog/unity/paint_balls/Assets/Media/armsBottom3.jpg',
        '/home/gilcu2/prog/unity/paint_balls/Assets/Media/armsBottom4.jpg',
        '/home/gilcu2/prog/unity/paint_balls/Assets/Media/armsBottom5.jpg',
    ]

    arms_front_files = [
        '/home/gilcu2/prog/unity/paint_balls/Assets/Media/armsFront1.png',
        '/home/gilcu2/prog/unity/paint_balls/Assets/Media/armsFront3.jpg',
        '/home/gilcu2/prog/unity/paint_balls/Assets/Media/armsFront4.jpg',
        '/home/gilcu2/prog/unity/paint_balls/Assets/Media/armsFront5.jpg',
    ]

    arms_open_files = [
        '/home/gilcu2/prog/unity/paint_balls/Assets/Media/armsOpen1.png',
        '/home/gilcu2/prog/unity/paint_balls/Assets/Media/armsOpen3.jpg',
        '/home/gilcu2/prog/unity/paint_balls/Assets/Media/armsOpen4.jpg',
        '/home/gilcu2/prog/unity/paint_balls/Assets/Media/armsOpen5.jpg',
    ]

    arms_top_files = [
        '/home/gilcu2/prog/unity/paint_balls/Assets/Media/armsTop1.png',
        '/home/gilcu2/prog/unity/paint_balls/Assets/Media/armsTop3.jpg',
        '/home/gilcu2/prog/unity/paint_balls/Assets/Media/armsTop4.jpg',
        '/home/gilcu2/prog/unity/paint_balls/Assets/Media/armsTop5.jpg',
    ]

    pose(arms_top_files)
