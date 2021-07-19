import cv2

capturaDeImagem = cv2.VideoCapture("VideosTeste/CaminhadaNaPraia.mp4")

while True:
    ret, frame = capturaDeImagem.read()
    cv2.imshow("Captura", frame)
    
    if cv2.waitKey(30) == 27: #Se a tecla esc for pressionada (Tecla 27), o video será fechado.
        break
    
capturaDeImagem.release()
cv2.destroyAllWindows()