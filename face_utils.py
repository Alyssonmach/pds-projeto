import numpy as np
import dlib
import cv2 

def facial_detection(gray_image, face_cascade):
    '''
    Múltipla detecção facial em imagem.

    Inputs:
        gray_image (array) -> Imagem para realização da inferência.
        face_cascade (CascadeClassifier) -> Arquitetura pré-treinada de detecção facial
    
    Returns:
        faces (list) -> Coordenadas para desenho da caixa delimitadora.
    '''

    # Detecta as faces na imagem utilizando a arquitetura treinada
    faces = face_cascade.detectMultiScale(image = gray_image, scaleFactor = 1.3, minNeighbors = 5,
                                          minSize = (30, 30))
    
    return faces 

def face_landmarks(gray_image, faces, predictor):
    '''
    Realiza a predição dos pontos facias a partir da caixa delimitadora.

    Inputs:
        gray_image (array) -> Imagem para realização da inferência.
        faces (list) -> Coordenadas para desenho da caixa delimitadora.
        predictor (shape_predictor) -> Arquitetura de predição de pontos faciais.
    
    Return:
        facial_points_np (array) -> Pontos faciais obtidos.
    '''
    
    # Converte as coordenadas para o padrão do dlib
    (x, y, w, h) = faces[0]
    face_dlib = dlib.rectangle(x, y, x + w, y + h)
    # Realiza a predição dos pontos faciais com o modelo de inferência
    facial_points = predictor(gray_image, face_dlib)
    # Converte os pontos faciais para um array numpy
    facial_points_np = np.array([(p.x, p.y) for p in facial_points.parts()])

    return facial_points_np

def get_facial_analisis_coords(points_ref):
    '''
    Obtém as caixas delimitadoras das bochechas e da testa através de pontos chave.

    Inputs:
        points_ref (list) -> Lista contendo os pontos faciais de referência.
    
    Returns:
        coords (list) -> Lista com as informações das coordenadas das caixas delimitadoras.
    '''

    # Extrai os pontos das bochechas e da testa
    forehead1, forehead2 = points_ref[0], points_ref[1]
    cheek1, cheek2 = points_ref[2], points_ref[3]

    # Delimita as coordenadas das caixas delimitadoras para cada regiaão
    coords_forehead = [forehead1[0], forehead1[1] - 40, forehead2[0], forehead2[1] - 10]
    coords_check1 = [cheek1[0] - 20, cheek1[1] + 25, cheek1[0] + 20, cheek1[1] + 50]
    coords_check2 = [cheek2[0] - 20, cheek2[1] + 25, cheek2[0] + 20, cheek2[1] + 50]

    # Adiciona as coordenadas em uma lista
    coords = [coords_forehead, coords_check1, coords_check2]

    return coords 

def extract_local_regions(image, coords_facial_locals):
    '''
    Extrai as regiões da imagem selecionadas em subimagens.

    Inputs:
        image (array) -> Imagem para realização da inferência.
        coords_facial_locals (list) -> Coordenadas das caixas delimitadoras dos pontos faciais.

    Returns:
        regions (list) -> Lista com as subimagens recortadas.

    '''
    
    # Inicializa a lista para adicionar as regiões faciais recortadas
    regions = list()

    for coords in coords_facial_locals:
        # Normaliza o padrão das coordenadas da imagem
        (x1, y1, x2, y2) = coords
        x, y = min(x1, x2), min(y1, y2)
        w, h = abs(x2 - x1), abs(y2 - y1)

        # Realiza o recorte das subimagens na imagem de referência e adiciona na lista
        x, y, w, h = int(x), int(y), int(w), int(h)
        region = image[y : y + h, x : x + w]
        regions.append(region)
    
    return regions
    
def draw_rectangle_face(image, faces):
    '''
    Desenha uma caixa delimitadora sobre a face detectada e informa o status.

    Inputs:
        image (array) -> Imagem para realização da inferência.
        faces (list) -> Coordenadas para desenho da caixa delimitadora.
    
        Returns:
            detection_flags (int) -> Informação sobre o status da detecção facial.
    '''

    # Se nenhuma face for detectada, retorna flag 0 
    if len(faces) == 0: return 0 

    # Se apenas uma face for detectada, retorna flag 1
    elif len(faces) == 1:
        # Desenha as informação de caixa delimitadora no frame.
        for (x, y, w, h) in faces:
            cv2.rectangle(img = image, pt1 = (x - 10, y - 50), pt2 = (x + w + 10, y + h + 30), 
                          color = (0, 0, 0), thickness = 2)
        
        return 1
    
    # Se mais de uma face for detectada, retorna flag 2
    else: return 2

def draw_landmarks(image, facial_points):
    '''
    Desenha na imagem os pontos faciais detectados.

    Inputs:
        image (array) -> Imagem para realização da inferência.
        facial_points (array) -> Pontos faciais obtidos.
    
    Returns:
        point_ref (list) -> Lista com os pontos de referência para a bochecha e da testa
    '''
    
    # Obtém os pontos de referência para análise do batimento cardíaco
    point_ref = list()
    # Desenhar um círculo em cada ponto facial
    for index, (x, y) in enumerate(facial_points):
        cv2.circle(img = image, center = (x, y), radius = 1, color = (0, 0, 255),
                thickness = cv2.FILLED)  
        # Obtém os pontos de referência para extrair fragmentos de imagem da bochecha e da testa
        if index == 19 or index == 24 or index == 41 or index == 46: 
            point_ref.append((x, y))
    
    return point_ref

def draw_rectangle_facial_locals(image, coords_facial_locals):
    '''
    Desenha o retângulo nos pontos faciais de análise para determinação do batimento cardiaco.

    Inputs:
        image (array) -> Imagem para realização da inferência.
        coords_facial_locals (list) -> Coordenadas das caixas delimitadoras dos pontos faciais.
    
    Returns:
    landmarks_flag (bool) -> Informa se todos os 3 locais faciais foram identificados.
    '''

    # Desenha caixas delimitadoras das bochechas e da testa
    if len(coords_facial_locals) == 3:

        cv2.rectangle(img = image, pt1 = (coords_facial_locals[0][0], coords_facial_locals[0][1]), 
                    pt2 = (coords_facial_locals[0][2], coords_facial_locals[0][3]), 
                    color = [0, 255, 0], thickness = 1)
        cv2.rectangle(img = image, pt1 = (coords_facial_locals[1][0], coords_facial_locals[1][1]), 
                    pt2 = (coords_facial_locals[1][2], coords_facial_locals[1][3]), 
                    color = [0, 255, 0], thickness = 1)
        cv2.rectangle(img = image, pt1 = (coords_facial_locals[2][0], coords_facial_locals[2][1]), 
                    pt2 = (coords_facial_locals[2][2], coords_facial_locals[2][3]), 
                    color = [0, 255, 0], thickness = 1)
        
        return 1
    
    # Caso haja alguma incoerência com os pontos das caixas delimitadoras, retorna falso
    else: return 0