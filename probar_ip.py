import cv2

# --- ¡CAMBIA ESTA LÍNEA con la IP y Puerto que ves en tu celular! ---
url_camara = "http://192.168.1.39:4747/video" 

print(f"Intentando conectar a: {url_camara}")
cap = cv2.VideoCapture(url_camara)

if not cap.isOpened():
    print("\n¡Error! No se pudo abrir el stream de video.")
    print("Verifica la URL y asegúrate de que el celular y la PC estén en la misma red Wi-Fi.")
else:
    print("\n¡Conexión exitosa! Presiona 'q' para salir.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Se perdió el stream.")
            break
        
        cv2.imshow("Prueba de Camara por IP", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()