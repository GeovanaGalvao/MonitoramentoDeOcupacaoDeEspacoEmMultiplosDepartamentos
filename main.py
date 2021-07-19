import cv2

capturaDeImagem = cv2.VideoCapture("VideosTeste/CaminhadaNaPraia.mp4")
detectorDeObjetos = cv2.createBackgroundSubtractorMOG2()
# Transforma a imagem para obter o resultado desejado.

while True:
    ret, frame = capturaDeImagem.read() #O frame obtem a imagem da camera, em quanto o ret
    #é uma variável booleana que retorna true caso o frame consiga obter a imagem da camera.
    mascara = detectorDeObjetos.apply(frame) #Aplica a mascara na foto da camera.   
    contorno, _ = cv2.findContours(mascara, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #Encontra
    #os objetos que estão em movimento e passa para a variável contorno. O segundo parametro não
    #erá utilizado, por isso o nome _ .
    
    for cnt in contorno:
        if cv2.contourArea(cnt) > 1000: #Ignora movimento de pequenos objetos
            cv2.drawContours(frame, [cnt], -1, (0, 200, 0), 2) #Desenha o contorno em volta dos
            #objetos em movimento. Os valores dentro do parenteses são paramentros do contorno.
    
    cv2.imshow("Captura", frame)
    #Abre uma caixa com o nome "Captura", mostrando a captura de imagem da camera.
    cv2.imshow("CapturaAlterada", mascara)
    
    if cv2.waitKey(30) == 27: #Se a tecla esc for pressionada, o video será fechado.
        break
    
capturaDeImagem.release() #Fecha o arquivo de video ou a captura da camera.
cv2.destroyAllWindows() #Essa linha garante que a memória vai ser realocada após o seu uso.