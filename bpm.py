import numpy as np
import cv2

class BPMFourier:

    def __init__(self, width, height) -> None:
        
        self.width = width
        self.height = height
        self.videoFrameRate = 60

        # Color Magnification Parameters
        self.levels = 3
        self.alpha = 170
        self.minFrequency = 1.0
        self.maxFrequency = 2.0
        self.bufferSize = 150
        self.bufferIndex = 0

        # Initialize Gaussian Pyramid
        self.firstFrame = np.zeros((self.height, self.width, 3))
        self.firstGauss = self.buildGauss(self.firstFrame, self.levels+1)[self.levels]
        self.videoGauss = np.zeros((self.bufferSize, self.firstGauss.shape[0], self.firstGauss.shape[1], 3))
        self.fourierTransformAvg = np.zeros((self.bufferSize))

        # Bandpass Filter for Specified Frequencies
        self.frequencies = (1.0*self.videoFrameRate) * np.arange(self.bufferSize) / (1.0*self.bufferSize)
        self.mask = (self.frequencies >= self.minFrequency) & (self.frequencies <= self.maxFrequency)

        # Heart Rate Calculation Variables
        self.bpmCalculationFrequency = 15
        self.bpmBufferIndex = 0
        self.bpmBufferSize = 10
        self.bpmBuffer = np.zeros((self.bpmBufferSize))

        self.i = 0

    # Helper Methods
    def buildGauss(self, frame, levels):
        pyramid = [frame]
        for level in range(levels):
            frame = cv2.pyrDown(frame)
            pyramid.append(frame)
        return pyramid
    def reconstructFrame(self, pyramid, index, levels):
        filteredFrame = pyramid[index]
        for level in range(levels):
            filteredFrame = cv2.pyrUp(filteredFrame)
        filteredFrame = filteredFrame[:self.height, :self.width]
        return filteredFrame
    
    def update(self, frame):

        self.videoGauss[self.bufferIndex] = self.buildGauss(frame, self.levels+1)[self.levels]
        fourierTransform = np.fft.fft(self.videoGauss, axis = 0)

        # Bandpass Filter
        fourierTransform[self.mask == False] = 0

        # Grab a Pulse
        if self.bufferIndex % self.bpmCalculationFrequency == 0:
            self.i = self.i + 1
            for buf in range(self.bufferSize):
                self.fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
            hz = self.frequencies[np.argmax(self.fourierTransformAvg)]
            bpm = 60.0 * hz
            self.bpmBuffer[self.bpmBufferIndex] = bpm
            self.bpmBufferIndex = (self.bpmBufferIndex + 1) % self.bpmBufferSize
        
        # Amplify
        filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
        filtered = filtered * self.alpha

        # Reconstruct Resulting Frame
        filteredFrame = self.reconstructFrame(filtered, self.bufferIndex, self.levels)
        outputFrame = frame + filteredFrame
        outputFrame = cv2.convertScaleAbs(outputFrame)
        
        self.bufferIndex = (self.bufferIndex + 1) % self.bufferSize

        if self.i > self.bpmBufferSize:
            bpm_data = self.bpmBuffer.mean()
        else:
            bpm_data = None
        
        return outputFrame, bpm_data
