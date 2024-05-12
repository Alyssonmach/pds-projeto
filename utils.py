import datetime
import cv2

def get_data_and_hour():
    '''
    Obtém uma string com data e horário para plotar na câmera.

    Outputs:
        date_and_hour (str) -> string com as informações de data e hora.
    '''

    actual_date = datetime.datetime.now().strftime('%d/%m/%Y')
    actual_hour = datetime.datetime.now().strftime('%H:%M')

    return f'[{actual_date}] - {actual_hour}'

def text_image(image, detection_flag):
    '''
    Desenha informações textuais sobre o frame de saída.

    Inputs:
        image (array) -> Imagem para realização da inferência.
        detection_flag (int) -> Informação do status da face detectada.
    '''

    # Adicionando o título do projeto no frame do vídeo
    cv2.putText(img = image, text = 'PDS - UFCG', org = (5, 20), 
                fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = [0, 255, 0],
                thickness = 1)
    # Adicionando a informação de data e hora atual
    cv2.putText(img = image, text = get_data_and_hour(), org = (440, 20), 
                fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = [0, 255, 0],
                thickness = 1)
    
    # Adiciona o status da quantidade de faces no frame
    if detection_flag == 0: face_info = 'Nenhuma Face Detectada'
    elif detection_flag == 1: face_info = 'Face Detectada'
    else: face_info = 'Multiplas faces detectadas'

    # Adicionando a informação de quantidade de faces no frame
    cv2.putText(img = image, text = f'Status: {face_info}', org = (5, 40), 
                fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = [0, 255, 0],
                thickness = 1)

