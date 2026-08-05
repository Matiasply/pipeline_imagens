import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# Inicializa o MediaPipe Face Mesh com as configurações desejadas
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, # Permite o rastreamento contínuo em vídeo
    max_num_faces=1, # Número máximo de faces a serem detectadas
    refine_landmarks=True, # Refinar os pontos de referência para incluir detalhes como a íris
    min_detection_confidence=0.5, # Confiança mínima para a detecção de faces
    min_tracking_confidence=0.5 # Confiança mínima para o rastreamento de pontos de referência
)

video = cv2.VideoCapture("video.mp4")

while True:
    ret, frame = video.read()

    # Se não houver mais frames para ler, sai do loop
    if (not ret):
        break
    
    # Converte o frame de BGR para RGB, pois o MediaPipe espera imagens no formato RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processa o frame para detectar os pontos de referência da face
    results = face_mesh.process(rgb_frame)

    if (results.multi_face_landmarks):
        # Desenha os pontos de referência da face no frame original
        for face_landmarks in results.multi_face_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACE_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
                mp.solutions.drawing_styles.get_default_face_mesh_contour_style()
            )

    cv2.imshow("Face", frame)

    if (cv2.waitKey(30) == 27):
        break

video.release()
cv2.destroyAllWindows()