#include <Arduino.h>
#include <math.h> // For fmod, fmax, constrain, floor

// --- Configuration Constants ---
// Actuator & Pressure Settings
const int PRESSURE_PIN_1 = A0;
const int PRESSURE_PIN_2 = A1;
const int PRESSURE_PIN_3 = A15;

const int SOLENOID_PRESS_1 = 2;
const int SOLENOID_DEPRESS_1 = 3;
const int SOLENOID_PRESS_2 = 4;
const int SOLENOID_DEPRESS_2 = 5;
const int SOLENOID_PRESS_3 = 8;
const int SOLENOID_DEPRESS_3 = 9;
const int SOLENOID_RELIEF = 12; // <-- NEW: 7th Solenoid for pressure relief

// Pressure sensor calibration (MUST BE ADJUSTED for your specific sensors)
const float VOLTS_TO_KPA = 155.55;
const float KPA_OFFSET_1 = -107.0;
const float KPA_OFFSET_2 = -111.0;
const float KPA_OFFSET_3 = -210.0;

const float MIN_OPERATING_PRESSURE_KPA = 5.0;   // Absolute minimum operating pressure
const float MAX_OPERATING_PRESSURE_KPA = 30.0;  // Absolute maximum operating pressure
// PRESSURE_RANGE_KPA is now calculated dynamically based on altitude floor

// Control Settings
const float HYSTERESIS_KPA = 2.0; // Deadband for pressure regulation

// --- Global Variables ---
// Target orientation from Serial input
float targetAzimuth = 0.0;   // Target Azimuth (0-360 degrees)
float targetAltitude = 90.0; // Target Altitude (0-90 degrees, 0=straight up, 90=horizontal)

// Pressure control variables
float feedforwardPressure1 = MIN_OPERATING_PRESSURE_KPA;
float feedforwardPressure2 = MIN_OPERATING_PRESSURE_KPA;
float feedforwardPressure3 = MIN_OPERATING_PRESSURE_KPA;

bool targetReached1 = false, targetReached2 = false, targetReached3 = false;
float prevTargetP1 = -1, prevTargetP2 = -1, prevTargetP3 = -1; // For detecting changes in target pressure

// Data Logging
bool headerPrinted = false;
unsigned long collectionStart = 0;

// --- Function Declarations ---
void setupPins();
void handleSerialInput();
void calculateRevisedFeedforwardPressures(float targetAzi, float targetAlti, float &p1, float &p2, float &p3);
void regulatePressure(float currentPressure, int pressPin, int depressPin, bool &targetReachedFlag, float targetPressureVal);
float readPressureSensor(int pin, float offset);
void printFeedforwardData(float actualP1, float actualP2, float actualP3);
float normalizeAngle(float angle);
void controlReliefSolenoid(); // <-- NEW: Function to control the relief solenoid

// --- Setup ---
void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);
  Serial.println("Revised Feedforward Actuator Control Initializing (with Relief Solenoid)...");

  setupPins();
  collectionStart = millis();
  Serial.println("Initialization Complete. Enter Target Azimuth (0-360) and Altitude (0-90) separated by a comma (e.g., 45,60)");
}

// --- Main Loop ---
void loop() {
  handleSerialInput(); // Check for new target angles from Serial

  // Calculate the ideal feedforward pressures using the revised logic
  calculateRevisedFeedforwardPressures(targetAzimuth, targetAltitude, feedforwardPressure1, feedforwardPressure2, feedforwardPressure3);

  // Detect if target pressures changed to reset hysteresis flags
  if (feedforwardPressure1 != prevTargetP1 || feedforwardPressure2 != prevTargetP2 || feedforwardPressure3 != prevTargetP3) {
      targetReached1 = targetReached2 = targetReached3 = false;
      prevTargetP1 = feedforwardPressure1;
      prevTargetP2 = feedforwardPressure2;
      prevTargetP3 = feedforwardPressure3;
  }

  // Read current actual pressures
  float currentPressure1 = readPressureSensor(PRESSURE_PIN_1, KPA_OFFSET_1);
  float currentPressure2 = readPressureSensor(PRESSURE_PIN_2, KPA_OFFSET_2);
  float currentPressure3 = readPressureSensor(PRESSURE_PIN_3, KPA_OFFSET_3);

  // Regulate pressures towards the feedforward targets
  regulatePressure(currentPressure1, SOLENOID_PRESS_1, SOLENOID_DEPRESS_1, targetReached1, feedforwardPressure1);
  regulatePressure(currentPressure2, SOLENOID_PRESS_2, SOLENOID_DEPRESS_2, targetReached2, feedforwardPressure2);
  regulatePressure(currentPressure3, SOLENOID_PRESS_3, SOLENOID_DEPRESS_3, targetReached3, feedforwardPressure3);

  // --- NEW: Control the relief solenoid ---
  controlReliefSolenoid();

  // Print data periodically
  printFeedforwardData(currentPressure1, currentPressure2, currentPressure3);
  
  // delay(10); // Optional small delay if needed
}

// --- Function Implementations ---

void setupPins() {
  pinMode(SOLENOID_PRESS_1, OUTPUT);
  pinMode(SOLENOID_DEPRESS_1, OUTPUT);
  pinMode(SOLENOID_PRESS_2, OUTPUT);
  pinMode(SOLENOID_DEPRESS_2, OUTPUT);
  pinMode(SOLENOID_PRESS_3, OUTPUT);
  pinMode(SOLENOID_DEPRESS_3, OUTPUT);
  pinMode(SOLENOID_RELIEF, OUTPUT); // <-- NEW: Initialize relief solenoid pin

  digitalWrite(SOLENOID_PRESS_1, LOW);
  digitalWrite(SOLENOID_DEPRESS_1, LOW);
  digitalWrite(SOLENOID_PRESS_2, LOW);
  digitalWrite(SOLENOID_DEPRESS_2, LOW);
  digitalWrite(SOLENOID_PRESS_3, LOW);
  digitalWrite(SOLENOID_DEPRESS_3, LOW);
  digitalWrite(SOLENOID_RELIEF, LOW); // <-- NEW: Start with relief solenoid OFF

  pinMode(PRESSURE_PIN_1, INPUT);
  pinMode(PRESSURE_PIN_2, INPUT);
  pinMode(PRESSURE_PIN_3, INPUT);
  Serial.println("Pins Initialized.");
}

void handleSerialInput() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    int commaIndex = input.indexOf(',');
    if (commaIndex != -1) {
      String aziStr = input.substring(0, commaIndex);
      String altStr = input.substring(commaIndex + 1);

      float newAzi = aziStr.toFloat();
      float newAlt = altStr.toFloat();

      if (aziStr.length() > 0 && altStr.length() > 0) {
          targetAzimuth = constrain(newAzi, 0.0, 360.0);
          targetAltitude = constrain(newAlt, 0.0, 90.0);
          Serial.print("New Target Set -> Azimuth: ");
          Serial.print(targetAzimuth, 1);
          Serial.print(" deg, Altitude: ");
          Serial.print(targetAltitude, 1);
          Serial.println(" deg");
          // Reset pressure target reached flags when target changes
          targetReached1 = targetReached2 = targetReached3 = false;
      } else {
          Serial.println("Invalid input format. Use: azimuth,altitude (e.g., 45,60)");
      }
    } else {
        Serial.println("Invalid input format. Use: azimuth,altitude (e.g., 45,60)");
    }
    while(Serial.available() > 0) Serial.read(); // Clear buffer
  }
}

// Calculates the REVISED feedforward target pressures for each segment
void calculateRevisedFeedforwardPressures(float targetAzi, float targetAlti, float &p1, float &p2, float &p3) {
    // 1. Calculate Altitude Floor Pressure (P_alt_floor)
    float altFactor = (90.0 - constrain(targetAlti, 0.0, 90.0)) / 90.0; // 1 for 0deg alt, 0 for 90deg alt
    float altitudeFloorPressure = MIN_OPERATING_PRESSURE_KPA + altFactor * (MAX_OPERATING_PRESSURE_KPA - MIN_OPERATING_PRESSURE_KPA);

    // 2. Calculate Dynamic Pressure Range for Azimuth modulation
    float dynamicPressureRangeForAzimuth = MAX_OPERATING_PRESSURE_KPA - altitudeFloorPressure;
    // Ensure range is not negative if altitudeFloorPressure is at MAX_OPERATING_PRESSURE_KPA
    if (dynamicPressureRangeForAzimuth < 0) {
        dynamicPressureRangeForAzimuth = 0;
    }

    // 3. Determine Azimuth Scaling Factors (alpha_azi_k) for each segment (0 to 1)
    float aziScalingFactor1 = 0.0;
    float aziScalingFactor2 = 0.0;
    float aziScalingFactor3 = 0.0;
    
    float normalizedAzi = normalizeAngle(targetAzi);
    int sector = floor(normalizedAzi / 60.0);
    float pos = fmod(normalizedAzi, 60.0) / 60.0; // Position within the sector (0 to 1)

    switch (sector) {
        case 0: // 0-60° (Seg 1 active, Seg 2 increasing)
            aziScalingFactor1 = 1.0;
            aziScalingFactor2 = pos;
            aziScalingFactor3 = 0.0;
            break;
        case 1: // 60-120° (Seg 2 active, Seg 1 decreasing)
            aziScalingFactor1 = 1.0 - pos;
            aziScalingFactor2 = 1.0;
            aziScalingFactor3 = 0.0;
            break;
        case 2: // 120-180° (Seg 2 active, Seg 3 increasing)
            aziScalingFactor1 = 0.0;
            aziScalingFactor2 = 1.0;
            aziScalingFactor3 = pos;
            break;
        case 3: // 180-240° (Seg 3 active, Seg 2 decreasing)
            aziScalingFactor1 = 0.0;
            aziScalingFactor2 = 1.0 - pos;
            aziScalingFactor3 = 1.0;
            break;
        case 4: // 240-300° (Seg 3 active, Seg 1 increasing)
            aziScalingFactor1 = pos;
            aziScalingFactor2 = 0.0;
            aziScalingFactor3 = 1.0;
            break;
        case 5: // 300-360° (Seg 1 active, Seg 3 decreasing)
            aziScalingFactor1 = 1.0;
            aziScalingFactor2 = 0.0;
            aziScalingFactor3 = 1.0 - pos;
            break;
    }

    // 4. Calculate Final Target Pressures
    p1 = altitudeFloorPressure + aziScalingFactor1 * dynamicPressureRangeForAzimuth;
    p2 = altitudeFloorPressure + aziScalingFactor2 * dynamicPressureRangeForAzimuth;
    p3 = altitudeFloorPressure + aziScalingFactor3 * dynamicPressureRangeForAzimuth;

    // 5. Constrain Final Pressures (primarily to cap at MAX_OPERATING_PRESSURE_KPA)
    p1 = constrain(p1, MIN_OPERATING_PRESSURE_KPA, MAX_OPERATING_PRESSURE_KPA);
    p2 = constrain(p2, MIN_OPERATING_PRESSURE_KPA, MAX_OPERATING_PRESSURE_KPA);
    p3 = constrain(p3, MIN_OPERATING_PRESSURE_KPA, MAX_OPERATING_PRESSURE_KPA);
}

// Bang-bang pressure regulation with hysteresis
void regulatePressure(float currentPressure, int pressPin, int depressPin, bool &targetReachedFlag, float targetPressureVal) {
    float deadLow = targetPressureVal - HYSTERESIS_KPA / 2.0;
    float deadHigh = targetPressureVal + HYSTERESIS_KPA / 2.0;

    // Ensure targetPressureVal itself is within physical limits before setting deadband
    targetPressureVal = constrain(targetPressureVal, MIN_OPERATING_PRESSURE_KPA, MAX_OPERATING_PRESSURE_KPA);

    if (!targetReachedFlag) {
        if (currentPressure < deadLow) {
            digitalWrite(pressPin, HIGH);
            digitalWrite(depressPin, LOW);
        }
        else if (currentPressure > deadHigh) {
            digitalWrite(pressPin, LOW);
            digitalWrite(depressPin, HIGH);
        }
        else { // Within deadband
            digitalWrite(pressPin, LOW);
            digitalWrite(depressPin, LOW);
            targetReachedFlag = true;
        }
    } else { // Target was reached
        digitalWrite(pressPin, LOW);
        digitalWrite(depressPin, LOW);
        // If pressure drifts too far out of a slightly wider band, re-engage control
        if (currentPressure > deadHigh + HYSTERESIS_KPA * 0.25 || currentPressure < deadLow - HYSTERESIS_KPA * 0.25) {
            targetReachedFlag = false;
        }
    }
}

// Reads a pressure sensor and applies calibration
float readPressureSensor(int pin, float offset) {
  int rawValue = analogRead(pin);
  float voltage = rawValue * (5.0 / 1023.0);
  float pressure = voltage * VOLTS_TO_KPA + offset;
  return fmax(pressure, 0.0);
}

// Prints data to Serial Plotter / Monitor
void printFeedforwardData(float actualP1, float actualP2, float actualP3) {
  if (millis() - collectionStart < 1000) return;

  if (!headerPrinted) {
    Serial.println("TargetAzi,TargetAlti,TargetP1,TargetP2,TargetP3,ActualP1,ActualP2,ActualP3");
    headerPrinted = true;
  }

  Serial.print(targetAzimuth, 1); Serial.print(',');
  Serial.print(targetAltitude, 1); Serial.print(',');
  Serial.print(feedforwardPressure1, 1); Serial.print(',');
  Serial.print(feedforwardPressure2, 1); Serial.print(',');
  Serial.print(feedforwardPressure3, 1); Serial.print(',');
  Serial.print(actualP1, 1); Serial.print(',');
  Serial.print(actualP2, 1); Serial.print(',');
  Serial.println(actualP3, 1);
}

// --- Helper Functions ---
// Normalize angle to 0-360 degrees
float normalizeAngle(float angle) {
    float result = fmod(angle, 360.0);
    if (result < 0) {
        result += 360.0;
    }
    return result;
}

// --- NEW: Function to control the relief solenoid ---
void controlReliefSolenoid() {
    // Check if pressurizing solenoids for actuators 1, 2, and 3 are all OFF
    // SOLENOID_PRESS_1 is pin 2
    // SOLENOID_PRESS_2 is pin 4
    // SOLENOID_PRESS_3 is pin 8
    bool press1_off = (digitalRead(SOLENOID_PRESS_1) == LOW);
    bool press2_off = (digitalRead(SOLENOID_PRESS_2) == LOW);
    bool press3_off = (digitalRead(SOLENOID_PRESS_3) == LOW);

    if (press1_off && press2_off && press3_off) {
        // If all three main pressurizing solenoids are OFF, turn ON the relief solenoid
        digitalWrite(SOLENOID_RELIEF, HIGH);
    } else {
        // Otherwise, turn OFF the relief solenoid
        digitalWrite(SOLENOID_RELIEF, LOW);
    }
}
