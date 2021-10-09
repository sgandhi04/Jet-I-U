#include <Arduino.h>
#include <Wire.h>
#include <SoftwareSerial.h>
#include <MeAuriga.h>
 
MeUltrasonicSensor ultrasonic_7(7);
MeBuzzer buzzer;
MeLightSensor lightsensor_12(12);
 
MeLEDMatrix ledMtx_9(9);
unsigned char drawBuffer[16];
unsigned char *drawTemp;
Servo servo_10_1;
MePort port_10(10);
Servo servo_10_2;
Me4Button buttonSensor_6(6);
MeDCMotor motor_1(1);
MeDCMotor motor_2(2);
Servo servo_9_1;
MePort port_9(9);
Servo servo_9_2;
void Flap (){
   servo_9_1.write(45);
   servo_9_2.write(100);
   _delay(0.5);
   servo_9_1.write(0);
   servo_9_2.write(180);
   _delay(0.5);
 
}
 
void _delay(float seconds) {
 if(seconds < 0.0){
   seconds = 0.0;
 }
 long endTime = millis() + seconds * 1000;
 while(millis() < endTime) _loop();
}
 
void setup() {
 ledMtx_9.setColorIndex(1);
 ledMtx_9.setBrightness(6);
 servo_10_1.attach(port_10.pin1());
 servo_10_2.attach(port_10.pin2());
 buzzer.setpin(45);
 randomSeed((unsigned long)(lightsensor_12.read() * 123456));
 servo_9_1.attach(port_9.pin1());
 servo_9_2.attach(port_9.pin2());
 drawTemp = new unsigned char[16]{10000110100000101111111010000000001111100010101000111010010000001111111001000000000000001011111000000000001111100000001000111110};
 memcpy(drawBuffer,drawTemp,16);
 free(drawTemp);
 ledMtx_9.drawBitmap(0,0,16,drawBuffer);
 while(1) {
     if(ultrasonic_7.distanceCm() < 40){
         for(int count=0;count<2;count++){
             Flap();
         }
         servo_10_1.write(15);
         servo_10_2.write(15);
 
         buzzer.tone(1500, 1 * 1000);
 
     }else{
         servo_10_1.write(125);
         servo_10_2.write(125);
         if((buttonSensor_6.pressed()==1)){
             motor_1.run(-1*50/100.0*255);
             motor_2.run(-1*50/100.0*255);
             _delay(3.5);
             motor_1.run(1*0/100.0*255);
             motor_2.run(1*0/100.0*255);
 
         }
         if((buttonSensor_6.pressed()==2)){
             motor_2.run(1*50/100.0*255);
             motor_1.run(1*50/100.0*255);
             _delay(3.5);
             motor_1.run(1*0/100.0*255);
             motor_2.run(1*0/100.0*255);
 
         }
 
     }
 
     _loop();
 }
 
}
 
void _loop() {
 buttonSensor_6.pressed();
}
 
void loop() {
 _loop();
}
