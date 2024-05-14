import numpy as np
import cv2

class BPMFourier:

    def __init__(self, width, height) -> None:
        '''
        Construtor da classe
        '''
        
        # Define a largura da imagem de entrada
        self.width = width
        # Define a altura da imagem de entrada
        self.height = height
        # FPS da aplicação (informação útil para determinação do filtro passa faixa)
        self.videoFrameRate = 60

        # Quantidade de níveis na pirâmide gaussiana
        self.levels = 3
        # Fator de amplificação da imagem filtrada
        self.alpha = 170
        # Frequência mínimas do filtro passa-banda
        self.minFrequency = 1.0
        # Frequência máxima do filtro passa-banda
        self.maxFrequency = 2.0
        # Tamanho do buffer para armazenamento dos quadros processados
        self.bufferSize = 150
        # Inicializando o índice do buffer
        self.bufferIndex = 0

        # Inicializando uma imagem em preto e branco com as dimensões da imagem de entrada
        self.firstFrame = np.zeros((self.height, self.width, 3))
        # Construindo a pirâmide gaussiana a partir da imagem inicial
        self.firstGauss = self.buildGauss(self.firstFrame, self.levels+1)[self.levels]
        # Criando um array multidimensional para armazenar os quadros da pirâmide gaussiana ao longo do tempo
        self.videoGauss = np.zeros((self.bufferSize, self.firstGauss.shape[0], self.firstGauss.shape[1], 3))
        # Inicializando um array para armazenar a média da transformada de Fourier ao longo do tempo
        self.fourierTransformAvg = np.zeros((self.bufferSize))

        # Sequência de frequências normalizadas
        self.frequencies = (1.0 * self.videoFrameRate) * np.arange(self.bufferSize) / (1.0 * self.bufferSize)
        # Máscara booleana que filtra as frequências da imagem dentro da banda especifificada (filtro passa-banda)
        self.mask = (self.frequencies >= self.minFrequency) & (self.frequencies <= self.maxFrequency)

        # Frequência na qual o cálculo do batimento cardíaco é realizada
        self.bpmCalculationFrequency = 60
        # Índice que controla a posição atual de armazenamento no buffer utilizado para calcular e armazenar os valores de BPM ao longo do tempo
        self.bpmBufferIndex = 0
        # Tamanho do buffer utilizado para armazenar os valores calculados de BPM ao longo do tempo.
        self.bpmBufferSize = 10
        # Variável para armazenar os valores calculados de BPM ao longo do tempo
        self.bpmBuffer = np.zeros((self.bpmBufferSize))

        # Contador que acompanha o número de quadros processados
        self.i = 0
        
    def buildGauss(self, frame, levels):
        '''
        Constrói uma pirâmide gaussiana a partir de um quadro de imagem.

        Inputs:
           frame (array) -> quadro de imagem de entrada.
           levels (int) -> número de níveis desejados da pirâmide gaussiana.
        
        Returns:
            pyramid (list) -> Lista contendo os níveis da pirâmide gaussiana, onde o primeiro elemento
        '''

        # Inicializa a lista da pirâmide com o quadro original
        pyramid = [frame]
        # Itera sobre o número de níveis desejados na pirâmide
        for level in range(levels):
            # Reduz o tamanho do quadro utilizando a operação de pirâmide Gaussiana (pyrDown)
            frame = cv2.pyrDown(frame)
            # Adiciona o quadro reduzido à lista da pirâmide
            pyramid.append(frame)
        
        # Retorna a lista completa da pirâmide gaussiana
        return pyramid
    
    def reconstructFrame(self, pyramid, index, levels):
        '''
        Reconstrói um quadro de imagem a partir de uma pirâmide gaussiana.

        Inputs:
            pyramid (list) -> Lista contendo os níveis da pirâmide gaussiana, onde cada elemento é um quadro de imagem.
            index (int) -> Índice do quadro na pirâmide a ser reconstruído.
            levels (int) -> Número de níveis utilizados na construção da pirâmide.
        
        Returns:
            filteredFrame (array) -> O quadro de imagem reconstruído a partir da pirâmide gaussiana
        '''

        # Seleciona o quadro na posição especificada pelo índice
        filteredFrame = pyramid[index]

        # Aplica a operação de pirâmide Gaussiana inversa (pyrUp) 'levels' vezes para reconstruir o quadro original
        for level in range(levels):
            filteredFrame = cv2.pyrUp(filteredFrame)
        
        # Recorta o quadro reconstruído para as dimensões originais especificadas (altura x largura)
        filteredFrame = filteredFrame[:self.height, :self.width]
        return filteredFrame
    
    def update(self, frame):
        '''
        Atualiza o processamento do algoritmo para um novo quadro de imagem.

        Inputs:
            frame (array) -> O novo quadro da imagem a ser processado.
        
        Returns:
            outputFrame (array) -> Quadro de imagem processado.
            bpm_data (float) -> Valor do BPM calculado.
        '''

        # Constrói a pirâmide gaussiana e armazena o nível desejado na lista de vídeo gaussiano
        self.videoGauss[self.bufferIndex] = self.buildGauss(frame, self.levels+1)[self.levels]
        # Aplica a transformada de Fourier ao vídeo gaussiano ao longo do tempo
        fourierTransform = np.fft.fft(self.videoGauss, axis = 0)

        # Aplica o filtro passa-banda à transformada de Fourier
        fourierTransform[self.mask == False] = 0

        # Verifica se é necessário calcular o BPM neste quadro
        if self.bufferIndex % self.bpmCalculationFrequency == 0:
            # Incrementa o contador de iterações
            self.i = self.i + 1
            # Calcula a média das amplitudes reais da transformada de Fourier ao longo do tempo
            for buf in range(self.bufferSize):
                self.fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
            # Determina a frequência dominante (Hz) com maior amplitude na transformada de Fourier
            hz = self.frequencies[np.argmax(self.fourierTransformAvg)]
            # Calcula o BPM correspondente à frequência dominante
            bpm = 60.0 * hz
            # Armazena o valor do BPM calculado no buffer
            self.bpmBuffer[self.bpmBufferIndex] = bpm
            # Atualiza o índice do buffer circular para o próximo valor
            self.bpmBufferIndex = (self.bpmBufferIndex + 1) % self.bpmBufferSize
        
        # Calcula o sinal filtrado invertendo a transformada de Fourier
        filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
        # Amplifica o sinal filtrado pela constante alpha
        filtered = filtered * self.alpha

        # Reconstrói o quadro resultante a partir do sinal filtrado
        filteredFrame = self.reconstructFrame(filtered, self.bufferIndex, self.levels)
        # Adiciona o quadro original ao quadro resultante
        outputFrame = frame + filteredFrame
        # Converte o quadro resultante para o formato de 8 bits sem sinal (0-255)
        outputFrame = cv2.convertScaleAbs(outputFrame)
        
        # Atualiza o índice do buffer circular para o próximo quadro
        self.bufferIndex = (self.bufferIndex + 1) % self.bufferSize

        # Verifica se já passou do número máximo de iterações para calcular o BPM
        if self.i > self.bpmBufferSize:
            # Calcula a média dos valores de BPM armazenados no buffer
            bpm_data = self.bpmBuffer.mean()
        else:
            # Retorna None se o cálculo do BPM ainda não estiver completo# Retorna None se o cálculo do BPM ainda não estiver completo
            bpm_data = None
        
        return outputFrame, bpm_data