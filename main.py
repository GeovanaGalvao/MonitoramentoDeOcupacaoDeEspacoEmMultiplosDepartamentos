import cv2

captura_de_imagem = cv2.VideoCapture("/home/pi/Desktop/VideosTeste/CameraLoja.mp4")
detector_de_objetos = cv2.createBackgroundSubtractorMOG2()
# Transforma a imagem para obter o resultado desejado.



while True:
    ret, frame = captura_de_imagem.read() #O frame obtem a imagem da camera, em quanto o ret
    #é uma variável booleana que retorna true caso o frame consiga obter a imagem da camera.
    altura, largura, _= frame.shape
    #print(largura, altura)
    regiao_de_interesse = frame[340:720,0:1280]
    
    #Reconhecimento dos objetos
    mascara = detector_de_objetos.apply(frame)  
    contorno, _ = cv2.findContours(mascara, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #Calculo para selecionar os objetos desejados.
    for cnt in contorno:
        if cv2.contourArea(cnt) > 1500: #Considera apenas objetos maiores do que o tamanho indicado.
            cv2.drawContours(frame, [cnt], -1, (0, 200, 0), 2)
    
    cv2.imshow("Captura", frame)
    cv2.imshow("Captura da regiao de interesse", regiao_de_interesse)
    #cv2.imshow("Captura com mascara", mascara)
    
    if cv2.waitKey(30) == 27: #Se a tecla esc for pressionada, o video será fechado.
        break
    
captura_de_imagem.release() #Fecha o arquivo de video ou a captura da camera.
cv2.destroyAllWindows() #Essa linha garante que a memória vai ser realocada após o seu uso.