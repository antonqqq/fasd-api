import cv2
import mediapipe as mp
import imutils

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode = True,
    max_num_faces = 1,
    min_detection_confidence = 0.8
)

face = cv2.imread('images/face5.jpg')
# face = imutils.resize(face, width=800)
# face = cv2.resize(face, (2560, 2560))
height, width, _ = face.shape
rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

result = face_mesh.process(face)

# print(result.multi_face_landmarks)

for landmark in result.multi_face_landmarks:
    for i in range(0, 468):
        pt1 = landmark.landmark[i]
        x = int(pt1.x * width)
        y = int(pt1.y * height)
        
        cv2.circle(face, (x, y), 1, (100, 100, 0), -1)
        cv2.putText(face, str(i), (x, y), 0, 0.2, (0, 0, 0))

cv2.imshow("face", face)
cv2.waitKey(0)
cv2.imwrite("result.png", face)