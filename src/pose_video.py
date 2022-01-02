import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
BG_COLOR = (192, 192, 192)


# For webcam input:
def pose():
    cap = cv2.VideoCapture(4)
    idx = 0
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5, enable_segmentation=True) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            fliped_image=cv2.flip(image, 1)
            cv2.imwrite(f'/tmp/image{idx:04d}.png', fliped_image)
            idx += 1
            results = pose.process(image)

            # Draw segmentation on the image.
            # To improve segmentation around boundaries, consider applying a joint
            # bilateral filter to "results.segmentation_mask" with "image".
            annotated_image = image.copy()
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
            annotated_image = np.where(condition, annotated_image, bg_image)

            # Draw landmark annotation on the image.
            # image.flags.writeable = True
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles
                    .get_default_pose_landmarks_style())
            # mp_drawing.plot_landmarks(
            #     results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Pose', cv2.flip(annotated_image, 1))
            world_landmarks=results.pose_world_landmarks.landmark
            print(f'{world_landmarks[0]} {world_landmarks[24]} {world_landmarks[25]}')
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()


if __name__ == '__main__':
    pose()
