
import face_utils
import numpy as np
import utils
import dlib 
import cv2
import bpm

forehead_bpm = bpm.BPMFourier(width = 100, height = 40)
checks1_bpm = bpm.BPMFourier(width = 40, height = 25)
checks2_bpm = bpm.BPMFourier(width = 40, height = 25)

# Dispositivo de captura de entrada (0 -> dispositivo padrão de webcam)
cap = cv2.VideoCapture(0)

# Verificar se a câmera abriu corretamente
if not cap.isOpened():
    print('Erro para inicializar a entrada de vídeo.')
    exit()

# Pesos treinados do detector de faces (Local Binary Patterns Improved)
facial_detection_path = 'lib64/python3.10/site-packages/cv2/data/lbpcascade_frontalface_improved.xml'
# Carregando a arquitetura de detecção facial
face_cascade = cv2.CascadeClassifier(facial_detection_path)
# Pesos treinados do modelo de detecção de pontos faciais (Face Landmarks - 68 points)
predictor_landmarks = dlib.shape_predictor('lib64/python3.10/site-packages/dlib/shape_predictor_68_face_landmarks.dat')

# Intera sobre os frames da WebCam
while cap.isOpened():

    # Captura o frame atual da WebCam
    ret, frame = cap.read()
    # Normaliza as dimensões do display de vídeo
    frame = cv2.resize(src = frame, dsize = (640, 480))
    # Espelha a imagem horizontalmente para correção de visualização
    frame = cv2.flip(src = frame, flipCode = 1)
    # Cria uma cópia da imagem de referência pré-processada
    frame_copy = frame.copy()
    # Modifica a imagem para a escala de cinza em apenas 1 canal
    gray = cv2.cvtColor(src = frame, code = cv2.COLOR_BGR2GRAY)

    # Caso não consiga interar sobre o frame, finaliza a interação
    if not ret: 
        print('Erro na entrada de vídeo.')
        break
    
    # Realiza a detecção facial no frame atual com um modelo pré-treinado
    faces = face_utils.facial_detection(gray_image = gray, face_cascade = face_cascade)
    # Desenha as caixas delimitadoras sobre as faces detectadas
    info_detection = face_utils.draw_rectangle_face(image = frame, faces = faces)
    
    
    if len(faces) == 1:
        # Obtém os pontos facias a partir da caixa delimitadoras
        facial_points = face_utils.face_landmarks(gray_image = gray, faces = faces, 
                                                  predictor = predictor_landmarks)
        # Desenha na imagem os pontos facias e obtém os pontos de análise da bochecha e da testa
        points_ref = face_utils.draw_landmarks(image = frame, facial_points = facial_points)
        # Obtém as coordenadas das caixas delimitadoras das bochechas e da testa para análise
        coords_facial_locals = face_utils.get_facial_analisis_coords(points_ref)
        # Desenha na imagem as coordenadas das bochechas e da testa
        landmarks_flag = face_utils.draw_rectangle_facial_locals(image = frame, 
                                                                 coords_facial_locals = coords_facial_locals)
        
        if landmarks_flag:
            # Extrai as imagens das bochechas e da testa para análise do batimento cardiaco
            image_locals = face_utils.extract_local_regions(image = frame_copy, 
                                                            coords_facial_locals = coords_facial_locals)
            
            forehead = cv2.resize(image_locals[0], (100, 40))
            checks1 = cv2.resize(image_locals[1], (40, 25))
            checks2 = cv2.resize(image_locals[1], (40, 25)) 

            forehead_image, forehead_data = forehead_bpm.update(frame = forehead)
            checks1_image, checks1_data = checks1_bpm.update(frame = checks1)
            checks2_image, checks2_data =checks2_bpm.update(frame = checks2)

            if forehead_data == None or checks1_data == None or checks2_data == None:
                cv2.putText(img = frame, text = 'Batimento por Minuto: Calculando', org = (5, 460), 
                            fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = [0, 0, 255],
                            thickness = 2)
            elif forehead_data != None and checks1_data != None and checks2_data != None:
                bpm_mean = (forehead_data + checks1_data + checks2_data) / 3
                bpm_mean = np.round(bpm_mean, 2)
                cv2.putText(img = frame, text = f'Batimento por Minuto: {bpm_mean}', org = (5, 460), 
                            fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = [0, 0, 255],
                            thickness = 2)

    # Aplica informações textuais sobre o frame
    utils.text_image(image = frame, detection_flag = info_detection)

    # Mostra a imagem de saída em um display externo
    cv2.imshow(winname = 'Batimento Cardiaco - PDS Projeto', mat = frame)
    
    # Captura a entrada de tecla do usuário e adiciona um delay na interação dos frames
    key= cv2.waitKey(delay = 1)

    # Caso o usuário pressione a tecla q, feche o programa
    if key == ord('q'): break

# Fecha o display corretamente gerado pelo OpenCV
cap.release()
cv2.destroyAllWindows()