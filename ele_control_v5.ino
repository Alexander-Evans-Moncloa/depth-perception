#include <Arduino.h>
#include <Adafruit_BNO08x.h>
#include <Wire.h>
#include <math.h> // For mathematical functions

// --- Configuration Constants ---
// IMU Settings
#define BNO08X_RESET -1 

#ifdef FAST_MODE
  sh2_SensorId_t reportType = SH2_GYRO_INTEGRATED_RV;
  long reportIntervalUs = 2000; 
#else
  sh2_SensorId_t reportType = SH2_ARVR_STABILIZED_RV; 
  long reportIntervalUs = 5000; 
#endif

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

const float VOLTS_TO_KPA = 155.55; 
const float KPA_OFFSET_1 = -107.0; 
const float KPA_OFFSET_2 = -111.0; 
const float KPA_OFFSET_3 = -210.0; 

const float MIN_OPERATING_PRESSURE_KPA = 5.0;   
const float MAX_OPERATING_PRESSURE_KPA = 30.0;  

const float ACTUATOR_PARAM_A = 1.5; 

// Control Settings
const float HYSTERESIS_KPA = 2.0; 

// PI Constants (Tune these carefully!)
const float KP_AZIMUTH = 0.08;  // Proportional gain for Azimuth
const float KI_AZIMUTH = 0.02;  // Integral gain for Azimuth (start small)

const float KP_ALTITUDE = 0.15; // Proportional gain for Altitude
const float KI_ALTITUDE = 0.03; // Integral gain for Altitude (start small)

// Integral windup limits
const float MAX_AZIMUTH_INTEGRAL_EFFECT = 5.0; // Max pressure change from Azimuth I-term (kPa)
const float MAX_ALTITUDE_INTEGRAL_EFFECT = 5.0; // Max pressure change from Altitude I-term (kPa)


// --- Global Variables ---
Adafruit_BNO08x bno08x(BNO08X_RESET);
sh2_SensorValue_t sensorValue;

struct EulerAngles {
  float yaw;
  float pitch;
  float roll;
};
EulerAngles currentOrientation; 

float targetAzimuth = 0.0;   
float targetAltitude = 90.0; 

float currentAzimuth = 0.0;
float currentAltitude = 90.0;

// PI variables
float azimuthError = 0.0, azimuthIntegral = 0.0;
float altitudeError = 0.0, altitudeIntegral = 0.0;
unsigned long lastPIDTime = 0;
float azimuthPIDOutput = 0.0;
float altitudePIDOutput = 0.0;

// Pressure control variables
float feedforwardP1 = MIN_OPERATING_PRESSURE_KPA;
float feedforwardP2 = MIN_OPERATING_PRESSURE_KPA;
float feedforwardP3 = MIN_OPERATING_PRESSURE_KPA;
float adjustedPressure1 = MIN_OPERATING_PRESSURE_KPA;
float adjustedPressure2 = MIN_OPERATING_PRESSURE_KPA;
float adjustedPressure3 = MIN_OPERATING_PRESSURE_KPA;

bool targetReached1 = false, targetReached2 = false, targetReached3 = false;
float prevAdjustedP1 = -1, prevAdjustedP2 = -1, prevAdjustedP3 = -1; 

bool headerPrinted = false;
unsigned long collectionStart = 0; 

// To store indices of segments based on their role in feedforward azimuth calc
int segIdx_MaxAziActivation = 1; // Segment most responsible for azimuth (e.g., factor 1.0)
int segIdx_MidAziActivation = 2; // Segment with intermediate factor (0 < factor < 1)
int segIdx_MinAziActivation = 3; // Segment with factor 0.0 for azimuth

// --- Function Declarations ---
void initialSetupIMU();
void setupPins();
void handleSerialInput();
void readIMU();
void calculateCurrentAzimuthAltitude();
void calculatePI_Outputs(); // Changed from calculatePID
void calculateRevisedFeedforwardPressures(float targetAzi, float targetAlti, float &ffP1, float &ffP2, float &ffP3, 
                                          int &idxMax, int &idxMid, int &idxMin); // Now also returns indices
void applyPI_ToPressures(float ffP1, float ffP2, float ffP3,
                         int idxMax, int idxMid, int idxMin,
                         float aziPI_Out, float altPI_Out,
                         float &adjP1, float &adjP2, float &adjP3);
void regulatePressure(float currentPressure, int pressPin, int depressPin, bool &targetReachedFlag, float targetPressure);
float readPressureSensor(int pin, float offset);
void printData(float actualP1, float actualP2, float actualP3);
float normalizeAngle(float angle); 
float angleDifference(float angle1, float angle2); 
void quaternionToEuler(float qr, float qi, float qj, float qk, EulerAngles* ypr, bool degrees);


// --- Setup ---
void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10); 
  Serial.println("Targeted PI Control System Initializing...");

  Wire.begin(); 
  setupPins();
  initialSetupIMU(); 

  lastPIDTime = millis();
  collectionStart = millis(); 
  Serial.println("Initialization Complete. Enter Target Azimuth (0-360) and Altitude (0-90) separated by a comma (e.g., 45,60)");
}

// --- Main Loop ---
void loop() {
  handleSerialInput(); 
  readIMU();           

  if (currentOrientation.pitch != -999) { 
    calculateCurrentAzimuthAltitude(); 
    calculatePI_Outputs(); // Calculate PI control outputs

    // Calculate feedforward pressures AND identify segment roles for azimuth
    calculateRevisedFeedforwardPressures(targetAzimuth, targetAltitude, 
                                         feedforwardP1, feedforwardP2, feedforwardP3,
                                         segIdx_MaxAziActivation, segIdx_MidAziActivation, segIdx_MinAziActivation);
    
    // Apply PI corrections to the specific segments
    applyPI_ToPressures(feedforwardP1, feedforwardP2, feedforwardP3,
                        segIdx_MaxAziActivation, segIdx_MidAziActivation, segIdx_MinAziActivation,
                        azimuthPIDOutput, altitudePIDOutput,
                        adjustedPressure1, adjustedPressure2, adjustedPressure3);

    // Detect if target pressures changed to reset hysteresis flags
    if (adjustedPressure1 != prevAdjustedP1 || adjustedPressure2 != prevAdjustedP2 || adjustedPressure3 != prevAdjustedP3) {
        targetReached1 = targetReached2 = targetReached3 = false;
        prevAdjustedP1 = adjustedPressure1;
        prevAdjustedP2 = adjustedPressure2;
        prevAdjustedP3 = adjustedPressure3;
    }

    float currentPressure1 = readPressureSensor(PRESSURE_PIN_1, KPA_OFFSET_1);
    float currentPressure2 = readPressureSensor(PRESSURE_PIN_2, KPA_OFFSET_2);
    float currentPressure3 = readPressureSensor(PRESSURE_PIN_3, KPA_OFFSET_3);

    regulatePressure(currentPressure1, SOLENOID_PRESS_1, SOLENOID_DEPRESS_1, targetReached1, adjustedPressure1);
    regulatePressure(currentPressure2, SOLENOID_PRESS_2, SOLENOID_DEPRESS_2, targetReached2, adjustedPressure2);
    regulatePressure(currentPressure3, SOLENOID_PRESS_3, SOLENOID_DEPRESS_3, targetReached3, adjustedPressure3);

    printData(currentPressure1, currentPressure2, currentPressure3);
  }
  // delay(1); // Optional
}

// --- Function Implementations ---

void setupPins() {
  pinMode(SOLENOID_PRESS_1, OUTPUT); pinMode(SOLENOID_DEPRESS_1, OUTPUT);
  pinMode(SOLENOID_PRESS_2, OUTPUT); pinMode(SOLENOID_DEPRESS_2, OUTPUT);
  pinMode(SOLENOID_PRESS_3, OUTPUT); pinMode(SOLENOID_DEPRESS_3, OUTPUT);

  digitalWrite(SOLENOID_PRESS_1, LOW); digitalWrite(SOLENOID_DEPRESS_1, LOW);
  digitalWrite(SOLENOID_PRESS_2, LOW); digitalWrite(SOLENOID_DEPRESS_2, LOW);
  digitalWrite(SOLENOID_PRESS_3, LOW); digitalWrite(SOLENOID_DEPRESS_3, LOW);

  pinMode(PRESSURE_PIN_1, INPUT); pinMode(PRESSURE_PIN_2, INPUT); pinMode(PRESSURE_PIN_3, INPUT);
  Serial.println("Pins Initialized.");
}

void initialSetupIMU() {
  if (!bno08x.begin_I2C()) {
    Serial.println("Failed to find BNO08x chip during initial setup.");
    while (1) { delay(10); } 
  }
  Serial.println("BNO08x Found!");
  Serial.print("Setting up report type: ");
  if (reportType == SH2_ARVR_STABILIZED_RV) Serial.println("ARVR Stabilized RV");
  else if (reportType == SH2_GYRO_INTEGRATED_RV) Serial.println("Gyro Integrated RV");

  if (!bno08x.enableReport(reportType, reportIntervalUs)) {
    Serial.println("Could not enable report during initial setup.");
  } else {
    Serial.println("Report enabled successfully during initial setup.");
  }
  delay(100); 
  currentOrientation.pitch = -999; 
}

void handleSerialInput() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n'); input.trim();
    int commaIndex = input.indexOf(',');
    if (commaIndex != -1) {
      String aziStr = input.substring(0, commaIndex); String altStr = input.substring(commaIndex + 1);
      float newAzi = aziStr.toFloat(); float newAlt = altStr.toFloat();
      if (aziStr.length() > 0 && altStr.length() > 0) { 
         targetAzimuth = constrain(newAzi, 0.0, 360.0);
         targetAltitude = constrain(newAlt, 0.0, 90.0); 
         Serial.print("New Target -> Azi: "); Serial.print(targetAzimuth, 1);
         Serial.print(" Alt: "); Serial.print(targetAltitude, 1); Serial.println(" deg");
         azimuthIntegral = 0.0; altitudeIntegral = 0.0; // Reset integrals on new target
         targetReached1 = targetReached2 = targetReached3 = false;
      } else { Serial.println("Invalid input. Format: azimuth,altitude"); }
    } else { Serial.println("Invalid input. Format: azimuth,altitude"); }
    while(Serial.available() > 0) Serial.read();
  }
}

void readIMU() {
  if (bno08x.wasReset()) {
    Serial.println("BNO08x reset! Re-enabling reports.");
    if (!bno08x.enableReport(reportType, reportIntervalUs)) {
      Serial.println("Failed to re-enable report. IMU unreliable.");
      currentOrientation.pitch = -999; 
    } else { Serial.println("Report re-enabled."); }
  }
  if (bno08x.getSensorEvent(&sensorValue)) {
    EulerAngles ypr_rad; 
    switch (sensorValue.sensorId) {
      case SH2_ARVR_STABILIZED_RV:
        quaternionToEuler(sensorValue.un.arvrStabilizedRV.real, sensorValue.un.arvrStabilizedRV.i, sensorValue.un.arvrStabilizedRV.j, sensorValue.un.arvrStabilizedRV.k, &ypr_rad, false); 
        break;
      case SH2_GYRO_INTEGRATED_RV:
        quaternionToEuler(sensorValue.un.gyroIntegratedRV.real, sensorValue.un.gyroIntegratedRV.i, sensorValue.un.gyroIntegratedRV.j, sensorValue.un.gyroIntegratedRV.k, &ypr_rad, false); 
        break;
      default: return; 
    }
    currentOrientation.yaw = ypr_rad.yaw * RAD_TO_DEG;
    currentOrientation.pitch = ypr_rad.pitch * RAD_TO_DEG;
    currentOrientation.roll = ypr_rad.roll * RAD_TO_DEG;
  }
}

void quaternionToEuler(float qr, float qi, float qj, float qk, EulerAngles* ypr, bool degrees) {
    float sqr = sq(qr); float sqi = sq(qi); float sqj = sq(qj); float sqk = sq(qk);
    float sum_sq = sqi + sqj + sqk + sqr;
    if (sum_sq < 1e-6) { ypr->yaw = 0; ypr->pitch = 0; ypr->roll = 0; return; } // Avoid division by zero
    ypr->yaw = atan2(2.0 * (qi * qj + qk * qr), (sqi - sqj - sqk + sqr));
    ypr->pitch = asin(-2.0 * (qi * qk - qj * qr) / sum_sq); 
    ypr->roll = atan2(2.0 * (qj * qk + qi * qr), (-sqi - sqj + sqk + sqr));
    if (degrees) { ypr->yaw *= RAD_TO_DEG; ypr->pitch *= RAD_TO_DEG; ypr->roll *= RAD_TO_DEG; }
}

void calculateCurrentAzimuthAltitude() {
  if (currentOrientation.pitch == -999) { currentAzimuth = 0; currentAltitude = 90; return; }
  float pitch_rad = currentOrientation.pitch * DEG_TO_RAD;
  float roll_rad = currentOrientation.roll * DEG_TO_RAD;
  currentAzimuth = atan2(sin(roll_rad), cos(roll_rad) * sin(pitch_rad)) * RAD_TO_DEG;
  currentAzimuth = normalizeAngle(currentAzimuth); 
  float cos_phi = constrain(ACTUATOR_PARAM_A * cos(roll_rad) * cos(pitch_rad), -1.0, 1.0); 
  currentAltitude = acos(cos_phi) * RAD_TO_DEG;
  currentAltitude = constrain(currentAltitude, 0.0, 90.0);
}

void calculatePI_Outputs() { // Renamed from calculatePID
    if (currentOrientation.pitch == -999) { 
        azimuthError = 0; altitudeError = 0; 
        azimuthPIDOutput = 0; altitudePIDOutput = 0;
        return; 
    }
    unsigned long now = millis();
    float timeDelta = (float)(now - lastPIDTime) / 1000.0; 

    if (timeDelta < 0.001) return; // Avoid too frequent updates / division by zero

    // Azimuth PI
    azimuthError = angleDifference(targetAzimuth, currentAzimuth); 
    azimuthIntegral += azimuthError * timeDelta;
    // Anti-windup for Azimuth Integral
    float max_azi_integral_val = MAX_AZIMUTH_INTEGRAL_EFFECT / (KI_AZIMUTH + 1e-6); // Max integral value before K_i scaling
    azimuthIntegral = constrain(azimuthIntegral, -max_azi_integral_val, max_azi_integral_val);
    azimuthPIDOutput = (KP_AZIMUTH * azimuthError) + (KI_AZIMUTH * azimuthIntegral);

    // Altitude PI
    altitudeError = targetAltitude - currentAltitude;
    altitudeIntegral += altitudeError * timeDelta;
    // Anti-windup for Altitude Integral
    float max_alt_integral_val = MAX_ALTITUDE_INTEGRAL_EFFECT / (KI_ALTITUDE + 1e-6);
    altitudeIntegral = constrain(altitudeIntegral, -max_alt_integral_val, max_alt_integral_val);
    altitudePIDOutput = (KP_ALTITUDE * altitudeError) + (KI_ALTITUDE * altitudeIntegral);
    
    lastPIDTime = now;
}

// Calculates REVISED feedforward pressures AND identifies segment roles for azimuth
void calculateRevisedFeedforwardPressures(float targetAzi, float targetAlti, 
                                          float &ffP1, float &ffP2, float &ffP3,
                                          int &idxMax, int &idxMid, int &idxMin) { // Output indices
    float altFactor = (90.0 - constrain(targetAlti, 0.0, 90.0)) / 90.0; 
    float altitudeFloorPressure = MIN_OPERATING_PRESSURE_KPA + altFactor * (MAX_OPERATING_PRESSURE_KPA - MIN_OPERATING_PRESSURE_KPA);
    float dynamicPressureRangeForAzimuth = MAX_OPERATING_PRESSURE_KPA - altitudeFloorPressure;
    if (dynamicPressureRangeForAzimuth < 0) dynamicPressureRangeForAzimuth = 0;

    float aziScalingFactor1 = 0.0, aziScalingFactor2 = 0.0, aziScalingFactor3 = 0.0;
    float normalizedAzi = normalizeAngle(targetAzi); 
    int sector = floor(normalizedAzi / 60.0);
    float pos = fmod(normalizedAzi, 60.0) / 60.0; 

    // Determine scaling factors and identify segment roles
    // This logic assumes segments are numbered 1, 2, 3
    // And azimuth progresses 1 -> 1&2 -> 2 -> 2&3 -> 3 -> 3&1
    switch (sector) {
        case 0: // Seg 1 max, Seg 2 mid, Seg 3 min
            aziScalingFactor1 = 1.0; idxMax = 1;
            aziScalingFactor2 = pos; idxMid = 2;
            aziScalingFactor3 = 0.0; idxMin = 3;
            break;
        case 1: // Seg 2 max, Seg 1 mid, Seg 3 min
            aziScalingFactor1 = 1.0 - pos; idxMid = 1;
            aziScalingFactor2 = 1.0;       idxMax = 2;
            aziScalingFactor3 = 0.0;       idxMin = 3;
            break;
        case 2: // Seg 2 max, Seg 3 mid, Seg 1 min
            aziScalingFactor1 = 0.0;       idxMin = 1;
            aziScalingFactor2 = 1.0;       idxMax = 2;
            aziScalingFactor3 = pos;       idxMid = 3;
            break;
        case 3: // Seg 3 max, Seg 2 mid, Seg 1 min
            aziScalingFactor1 = 0.0;       idxMin = 1;
            aziScalingFactor2 = 1.0 - pos; idxMid = 2;
            aziScalingFactor3 = 1.0;       idxMax = 3;
            break;
        case 4: // Seg 3 max, Seg 1 mid, Seg 2 min
            aziScalingFactor1 = pos;       idxMid = 1;
            aziScalingFactor2 = 0.0;       idxMin = 2;
            aziScalingFactor3 = 1.0;       idxMax = 3;
            break;
        case 5: // Seg 1 max, Seg 3 mid, Seg 2 min
            aziScalingFactor1 = 1.0;       idxMax = 1;
            aziScalingFactor2 = 0.0;       idxMin = 2;
            aziScalingFactor3 = 1.0 - pos; idxMid = 3;
            break;
    }

    ffP1 = altitudeFloorPressure + aziScalingFactor1 * dynamicPressureRangeForAzimuth;
    ffP2 = altitudeFloorPressure + aziScalingFactor2 * dynamicPressureRangeForAzimuth;
    ffP3 = altitudeFloorPressure + aziScalingFactor3 * dynamicPressureRangeForAzimuth;

    ffP1 = constrain(ffP1, MIN_OPERATING_PRESSURE_KPA, MAX_OPERATING_PRESSURE_KPA);
    ffP2 = constrain(ffP2, MIN_OPERATING_PRESSURE_KPA, MAX_OPERATING_PRESSURE_KPA);
    ffP3 = constrain(ffP3, MIN_OPERATING_PRESSURE_KPA, MAX_OPERATING_PRESSURE_KPA);
}


// Applies PI outputs to the specific segments based on their roles
void applyPI_ToPressures(float ffP1, float ffP2, float ffP3,
                         int idxMax, int idxMid, int idxMin,
                         float aziPI_Out, float altPI_Out,
                         float &adjP1, float &adjP2, float &adjP3) {
    // Initialize adjusted pressures with feedforward values
    adjP1 = ffP1;
    adjP2 = ffP2;
    adjP3 = ffP3;

    // Apply Azimuth PI to the "Mid" activation segment
    if (idxMid == 1) adjP1 += aziPI_Out;
    else if (idxMid == 2) adjP2 += aziPI_Out;
    else if (idxMid == 3) adjP3 += aziPI_Out;

    // Apply Altitude PI to the "Min" activation segment (which is effectively the altitude setter)
    if (idxMin == 1) adjP1 += altPI_Out;
    else if (idxMin == 2) adjP2 += altPI_Out;
    else if (idxMin == 3) adjP3 += altPI_Out;
    
    // The "Max" activation segment (idxMax) is not directly changed by PI here,
    // its pressure is taken from the feedforward calculation.

    // Constrain all pressures after PI adjustments
    adjP1 = constrain(adjP1, MIN_OPERATING_PRESSURE_KPA, MAX_OPERATING_PRESSURE_KPA);
    adjP2 = constrain(adjP2, MIN_OPERATING_PRESSURE_KPA, MAX_OPERATING_PRESSURE_KPA);
    adjP3 = constrain(adjP3, MIN_OPERATING_PRESSURE_KPA, MAX_OPERATING_PRESSURE_KPA);
}


void regulatePressure(float currentPressure, int pressPin, int depressPin, bool &targetReachedFlag, float targetPressureVal) {
    float deadLow = targetPressureVal - HYSTERESIS_KPA / 2.0;
    float deadHigh = targetPressureVal + HYSTERESIS_KPA / 2.0;
    targetPressureVal = constrain(targetPressureVal, MIN_OPERATING_PRESSURE_KPA, MAX_OPERATING_PRESSURE_KPA);

    if (!targetReachedFlag) {
        if (currentPressure < deadLow) { digitalWrite(pressPin, HIGH); digitalWrite(depressPin, LOW); }
        else if (currentPressure > deadHigh) { digitalWrite(pressPin, LOW); digitalWrite(depressPin, HIGH); }
        else { digitalWrite(pressPin, LOW); digitalWrite(depressPin, LOW); targetReachedFlag = true; }
    } else { 
        digitalWrite(pressPin, LOW); digitalWrite(depressPin, LOW);
        if (currentPressure > deadHigh + HYSTERESIS_KPA * 0.25 || currentPressure < deadLow - HYSTERESIS_KPA * 0.25) { 
             targetReachedFlag = false;
        }
    }
}

float readPressureSensor(int pin, float offset) {
  int rawValue = analogRead(pin);
  float voltage = rawValue * (5.0 / 1023.0); 
  float pressure = voltage * VOLTS_TO_KPA + offset;
  return fmax(pressure, 0.0); 
}

void printData(float actualP1, float actualP2, float actualP3) {
  if (millis() - collectionStart < 3000) return; 
  if (!headerPrinted) {
    Serial.println("TargetAzi,TargetAlti,CurrentAzi,CurrentAlti,AziError,AltiError,AziPI,AltPI,ffP1,ffP2,ffP3,adjP1,adjP2,adjP3,ActualP1,ActualP2,ActualP3,P,R");
    headerPrinted = true;
  }
  Serial.print(targetAzimuth, 1); Serial.print(','); Serial.print(targetAltitude, 1); Serial.print(',');
  Serial.print(currentAzimuth, 1); Serial.print(','); Serial.print(currentAltitude, 1); Serial.print(',');
  Serial.print(azimuthError, 1); Serial.print(','); Serial.print(altitudeError, 1); Serial.print(',');
  Serial.print(azimuthPIDOutput, 2); Serial.print(','); Serial.print(altitudePIDOutput, 2); Serial.print(',');
  Serial.print(feedforwardP1, 1); Serial.print(','); Serial.print(feedforwardP2, 1); Serial.print(','); Serial.print(feedforwardP3, 1); Serial.print(',');
  Serial.print(adjustedPressure1, 1); Serial.print(','); Serial.print(adjustedPressure2, 1); Serial.print(','); Serial.print(adjustedPressure3, 1); Serial.print(',');
  Serial.print(actualP1, 1); Serial.print(','); Serial.print(actualP2, 1); Serial.print(','); Serial.print(actualP3, 1); Serial.print(',');
  Serial.print(currentOrientation.pitch, 2); Serial.print(','); Serial.println(currentOrientation.roll, 2);
}

float normalizeAngle(float angle) {
    float result = fmod(angle, 360.0);
    if (result < 0) result += 360.0;
    return result;
}

float angleDifference(float angle1, float angle2) {
    float diff = normalizeAngle(angle1) - normalizeAngle(angle2);
    if (diff > 180.0) diff -= 360.0;
    else if (diff <= -180.0) diff += 360.0;
    return diff;
}
