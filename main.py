import cv2

captura_de_imagem = cv2.VideoCapture("/home/pi/Desktop/VideosTeste/CameraLoja.mp4")
detector_de_objetos = cv2.createBackgroundSubtractorMOG2(history = 100, varThreshold = 200)
# Transforma a imagem para obter o resultado desejado.

while True:
    ret, frame = captura_de_imagem.read() #O frame obtem a imagem da camera, em quanto o ret
    #é uma variável booleana que retorna true caso o frame consiga obter a imagem da camera.
    regiao_de_interesse = frame[340:720,0:1280]
    
    #Reconhecimento dos objetos
    mascara = detector_de_objetos.apply(regiao_de_interesse)
    _, mascara = cv2.threshold(mascara, 254, 255, cv2.THRESH_BINARY)
    contorno, _ = cv2.findContours(mascara, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #Calculo para selecionar os objetos desejados.
    for cnt in contorno:
        if cv2.contourArea(cnt) > 1500: #Considera apenas objetos maiores do que o tamanho indicado.
            #cv2.drawContours(regiao_de_interesse, [cnt], -1, (0, 200, 0), 2)            
            coordenada_x, coordenada_y, largura, altura = cv2.boundingRect(cnt)
            cv2.rectangle(regiao_de_interesse, (coordenada_x, coordenada_y), (coordenada_x + largura, coordenada_y + altura), (0, 255, 0), 3)
    
    cv2.imshow("Captura", frame)
    #cv2.imshow("Captura da regiao de interesse", regiao_de_interesse)
    #cv2.imshow("Captura com mascara", mascara)
    
    if cv2.waitKey(30) == 27: #Se a tecla esc for pressionada, o video será fechado.
        break
    
captura_de_imagem.release() #Fecha o arquivo de video ou a captura da camera.
cv2.destroyAllWindows() #Essa linha garante que a memória vai ser realocada após o seu uso.