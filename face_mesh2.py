import cv2
import mediapipe as mp
import imutils
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

index = 0
# For static images:
IMAGE_FILES = ['images/face6.jpg']
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.9) as face_mesh:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    # Convert the BGR image to RGB before processing.
    image = imutils.resize(image, width=4000)
    height, width, _ = image.shape
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print and draw face mesh landmarks on the image.
    if not results.multi_face_landmarks:
      continue
    annotated_image = image.copy()
    for face_landmarks in results.multi_face_landmarks:
      # print('face_landmarks:', face_landmarks)
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_TESSELATION,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_tesselation_style())
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_CONTOURS,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_contours_style())
      for i in range(0, 468):
        pt1 = face_landmarks.landmark[i]
        x = int(pt1.x * width)
        y = int(pt1.y * height)
        cv2.circle(annotated_image, (x, y), 2, (100, 100, 0), -1)
        cv2.putText(annotated_image, str(i), (x, y), 0, 0.4, (255, 255, 255))
    cv2.imwrite('annotated_image' + str(idx) + '.png', annotated_image)

# # For webcam input:
# drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
# cap = cv2.VideoCapture(0)
# with mp_face_mesh.FaceMesh(
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5) as face_mesh:
#   while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#       print("Ignoring empty camera frame.")
#       # If loading a video, use 'break' instead of 'continue'.
#       continue

#     # Flip the image horizontally for a later selfie-view display, and convert
#     # the BGR image to RGB.
#     image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     image.flags.writeable = False
#     results = face_mesh.process(image)

#     # Draw the face mesh annotations on the image.
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     if results.multi_face_landmarks:
#       for face_landmarks in results.multi_face_landmarks:
#         mp_drawing.draw_landmarks(
#             image=image,
#             landmark_list=face_landmarks,
#             connections=mp_face_mesh.FACEMESH_TESSELATION,
#             landmark_drawing_spec=None,
#             connection_drawing_spec=mp_drawing_styles
#             .get_default_face_mesh_tesselation_style())
#         mp_drawing.draw_landmarks(
#             image=image,
#             landmark_list=face_landmarks,
#             connections=mp_face_mesh.FACEMESH_CONTOURS,
#             landmark_drawing_spec=None,
#             connection_drawing_spec=mp_drawing_styles
#             .get_default_face_mesh_contours_style())
#     cv2.imshow('MediaPipe FaceMesh', image)
#     if cv2.waitKey(5) & 0xFF == 27:
#       break
# cap.release()