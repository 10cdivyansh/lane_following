#include <cvzone.h>
#include <AccelStepper.h>
SerialData serialData(1,3);
int valsRec[1];
AccelStepper stepper(1, 7, 6);
void setup(){
  serialData.begin(9600);
}
void move_stepper(int stp) {
  stepper.setMaxSpeed(400);
  stepper.setAcceleration(10000);
  stepper.moveTo(stp); 
  stepper.run();
}
void loop() {
  serialData.Get(valsRec);
  move_stepper(valsRec[0]);

}
