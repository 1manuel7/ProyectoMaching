// Incluimos la librería para controlar los servomotores
#include <Servo.h>

// Creamos los objetos para cada servomotor
Servo servoVerde;   // Servo para la categoría "verde"
Servo servoIntermedia; // Servo para la categoría "intermedia"
// No necesitamos un servo para "madura" si esa bandeja está al final

// --- CONFIGURACIÓN INICIAL ---
void setup() {
  // Iniciamos la comunicación con la computadora a 9600 baudios
  Serial.begin(9600);

  // Conectamos los servos a los pines digitales PWM
  // Asegúrate de que los pines coincidan con tus conexiones físicas
  servoVerde.attach(9);      // Conecta el servo 'verde' al pin 9
  servoIntermedia.attach(10); // Conecta el servo 'intermedia' al pin 10

  // Posición inicial de los servos (posición de reposo)
  servoVerde.write(0);
  servoIntermedia.write(0);
  
  // Mensaje de inicio para saber que el Arduino está listo
  Serial.println("Arduino listo para clasificar.");
}

// --- BUCLE PRINCIPAL ---
void loop() {
  // Si hay datos disponibles en el puerto serie...
  if (Serial.available() > 0) {
    // ...leemos el carácter que nos envía Python
    char comando = Serial.read();

    // --- Lógica de Clasificación ---
    if (comando == 'G') { // Si recibe 'G' (Verde)
      Serial.println("Recibido: Verde. Moviendo servo 1...");
      servoVerde.write(90);  // Mueve el servo a 90 grados
      delay(1000);           // Espera 1 segundo
      servoVerde.write(0);   // Vuelve a la posición inicial
    } 
    else if (comando == 'I') { // Si recibe 'I' (Intermedia)
      Serial.println("Recibido: Intermedia. Moviendo servo 2...");
      servoIntermedia.write(90); // Mueve el servo a 90 grados
      delay(1000);               // Espera 1 segundo
      servoIntermedia.write(0);   // Vuelve a la posición inicial
    }
    // Si recibe 'M' (Madura), no hacemos nada, la manzana sigue derecho.
  }
}
