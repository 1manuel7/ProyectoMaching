#include <Servo.h>

Servo servoVerde;
Servo servoIntermedia;

// --- CONSTANTES DE POSICIÓN ---
// Ajusta estos ángulos según tu prototipo
const int POS_REPOSO = 0;   // Posición donde el palito no bloquea nada
const int POS_DESVIO = 90; // Posición donde el palito desvía la manzana

void setup() {
  Serial.begin(9600); // Inicia comunicación con Python

  // Conecta los servos a los pines PWM
  servoVerde.attach(9);
  servoIntermedia.attach(10);

  // Asegurarse de que ambos servos empiecen en reposo
  servoVerde.write(POS_REPOSO);
  servoIntermedia.write(POS_REPOSO);
  
  Serial.println("Arduino listo para clasificar.");
}

void loop() {
  if (Serial.available() > 0) {
    char comando = Serial.read(); // Leer el comando de Python

    if (comando == 'V') { // 'V' para VERDE
      Serial.println("Recibido: VERDE. Moviendo servo 1...");
      servoVerde.write(POS_DESVIO);
      delay(1000); // Mantiene la posición por 1 segundo
      servoVerde.write(POS_REPOSO);
    } 
    else if (comando == 'I') { // 'I' para INTERMEDIA
      Serial.println("Recibido: INTERMEDIA. Moviendo servo 2...");
      servoIntermedia.write(POS_DESVIO);
      delay(1000); // Mantiene la posición por 1 segundo
      servoIntermedia.write(POS_REPOSO);
    }
    // Si el comando es 'M' (Madura), no hace nada.
    // La manzana sigue de frente y cae en la última caja.
  }
}
