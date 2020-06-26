/*
 * author : Georgiadis Thomas
 *
 * This project was created for the purpose of implmenting my master thesis for
 * Aristotle University of Thessaloniki
 *
 * Problem Description:
 *
 * The follow code drives a System that collects data from MPU6050 and processes them accordinately in order
 * to detect faulty operation in centrifigural pumps of certain type (Wilo Star RS 25/4). The Data is saved
 * in a .txt file inside the SD Card connected to the SD Card adapter.
 * This implementation is using FREERTOS for task scheduling making the system's time handling more robust and precise.
 * The data acquired from MPU6050 are processed using FFT in real time on 1 second period. The system is
 * gathering vibrations Frequency information for 10 seconds.Out of this information the mean peak frequencies of the sensor's
 * signal are extracted (x ~ y axis) .
 * The amplitude - DC component of the signal is also saved (0HZ frequency component of the spectrum).
 * This data is then fed to an offline trained Neural Network which is trained to detect faulty operations based on real data.
 *
 * 5/4/2020
 *
 * Added wifi connection feature and UTC local time get for better data logging. The time is acquired through ntp server connection
 * and requests.
 *
 * Added email sending feature. When the state of the centrifugal pump is critical,ESP32 sends and email from a test email to the
 * owner's email or the technician's email
 *
 *
 * 24/4/2020
 *
 * Change sampling frequency to 128 Hz.
 *
 *
 * 1/5/2020
 *
 * Change sampling frequency to 256 Hz.
 *
 *
 * 2/5/2020
 *
 * Changes sampling frequency to 512 Hz and correct data logging in SD card..
 *
 *
 */

#include "FS.h"
#include "SD.h"
#include "SPI.h"
#include <Wire.h>
#include "arduinoFFT.h"
#include "EloquentTinyML.h"
#include "Vibration_NN_model.h"

/* Libraries for internet connection ~ communication handling */
#include <WiFi.h>
#include "time.h"
#include "ESP32_MailClient.h"

/*
 * Definitions for TF Lite Model Handling feature.
 */

// NN input size.
#define INPUT_SIZE (4)
// NN output size.
#define OUTPUT_SIZE (1)
// Memory allocation size for input + output tensors
#define TENSOR_ARENA_SIZE (INPUT_SIZE+OUTPUT_SIZE)*1024
#define SAMPLES 512//256//128

// TF Lite object - Initialization of the NN model.
Eloquent::TinyML::TfLite<INPUT_SIZE, OUTPUT_SIZE, TENSOR_ARENA_SIZE> ml(Vibration_Stat_Regr_model_tflite);

// Array storing the peak frequencies (x~y) for 10 seconds -> 2*10 = 20.
float input[(INPUT_SIZE-2)*10];
// Array storing the NN inputs.
float input_nn[INPUT_SIZE];
// NN output.
float predicted = 0;

// Timer which decides when the system will start giving outputs(leds).
int output_timer = 0;

/* Create FFT object and variables */

// FFT object Initialization.
arduinoFFT FFT = arduinoFFT();
// FFT total samples.
const uint16_t samples = SAMPLES; //This value MUST ALWAYS be a power of 2
// FFT sampling Frequency -> 1/ts.
const double samplingFrequency = SAMPLES;

/* Vectors receiving the computed results from FFT*/
double vReal_x[SAMPLES];
double vReal_y[SAMPLES];
//double vReal_z[SAMPLES];

// Since the signal is Real, the Imaginary part is always going to be zero
// allowing the use of only one void array for the imaginary part of all the axis' (x-y-z).
double vImag[SAMPLES];

/* Vectors receiving the mean power of the signal from the DC component of FFT */
double ampl_x[10];
double ampl_y[10];

double idle_power_constant = 100;

int i = 0;
int j = 0;

/* Create MPU6050 object and Helping variables - definitions*/

// MPU6050 Slave Device Address
const uint8_t MPU6050SlaveAddress = 0x68;

/* Select SDA and SCL pins for I2C communication */
const uint8_t scl = 22;
const uint8_t sda = 21;

/* Sensitivity scale factor respective to full scale setting provided in datasheet */
const uint16_t AccelScaleFactor = 16384;
uint16_t  Ax_offset = -250;
uint16_t  Ay_offset =  36;

/* MPU6050 configuration register addresses */
const uint8_t MPU6050_REGISTER_SMPLRT_DIV          =  0x19; // Sample rate Divider
const uint8_t MPU6050_REGISTER_USER_CTRL           =  0x6A; // User preferences register
const uint8_t MPU6050_REGISTER_PWR_MGMT_1          =  0x6B; // Clock info /Device reset / Power mode register.
const uint8_t MPU6050_REGISTER_PWR_MGMT_2          =  0x6C; // On/Off of the sensor's modules.
const uint8_t MPU6050_REGISTER_CONFIG              =  0x1A; // Low pass filter and FSYNC
const uint8_t MPU6050_REGISTER_ACCEL_CONFIG        =  0x1C; // Accelerometer Configuration
const uint8_t MPU6050_REGISTER_FIFO_EN             =  0x23; // FIFO Enable
const uint8_t MPU6050_REGISTER_INT_ENABLE          =  0x38; // Interrupt register
const uint8_t MPU6050_REGISTER_ACCEL_XOUT_H        =  0x3B; // MSB of accelerometer reading
const uint8_t MPU6050_REGISTER_SIGNAL_PATH_RESET   =  0x68; // SIgnal path reset


/* MPU6050 Reading variables. */
int16_t AccelX, AccelY, AccelZ;
double Ax, Ay, Az;

/* SD Card read variables*/
char new_reading_char[100];
char *new_reading_char_ptr;

/* OutPut Leds Definitions */
const int led_green_Pin = 2; //GPIO2
const int led_red_Pin = 4;   //GPIO4

/* Network Credentials */

const char* ssid       = "anastasia_home";
const char* password   = "anastasia2016";

/* UTC Obtain Variables*/

// Europe pool for greece
const char* ntpServer = "europe.pool.ntp.org";
// UTC + 2 for greece
const long  gmtOffset_sec = 2*3600;
const int   daylightOffset_sec = 3600;

/* Email sending configuration. */

// Email data object
SMTPData smtpData;

// To send an Email using Gmail use port 465 (SSL) and SMTP Server smtp.gmail.com
// YOU MUST ENABLE less secure app option https://myaccount.google.com/lesssecureapps?pli=1

#define emailSenderAccount    "esp32TestThomas@gmail.com"
#define emailSenderPassword   "ESP32TESTSSS"
#define emailRecipient        "thomasgeorgiadis@ua.pt"
#define smtpServer            "smtp.gmail.com"
#define smtpServerPort        465
#define emailSubject          "Pump Notification Allert"

int email_enable = 1; // set to 1 at start

/* FREERTOS Tasks Initialization*/
TaskHandle_t ML_Task;
TaskHandle_t FFT_FUN_Task;
TaskHandle_t MPU6050_Task;
// TaskHandle_t SD_WRITE_Task

void ML_NN( void *pvParameters );
void FFT_FUN( void *pvParameters );
void MPU6050_READ( void *pvParameters );
//void SD_CARD_WRITE( void *pvParameters );

/* Setup Function that runs once while the program starts. */

void setup(){

    /* Initializing MPU6050 Handler ~ I2C Communication. */
    Wire.begin(sda, scl);
    MPU6050_Init();

    /* Initializing Serial Bus */
    Serial.begin(115200);

    /* Initializing SD Card */
    if(!SD.begin()){
        Serial.println("Card Mount Failed");
        return;
    }

    uint8_t cardType = SD.cardType();

    if(cardType == CARD_NONE){
        Serial.println("No SD card attached");
        return;
    }

    Serial.print("SD Card Type: ");
    if(cardType == CARD_MMC){
        Serial.println("MMC");
    } else if(cardType == CARD_SD){
        Serial.println("SDSC");
    } else if(cardType == CARD_SDHC){
        Serial.println("SDHC");
    } else {
        Serial.println("UNKNOWN");
    }

    uint64_t cardSize = SD.cardSize() / (1024 * 1024);
    Serial.printf("SD Card Size: %lluMB\n", cardSize);

    writeFile(SD, "/full_set_readings.txt", "Fx,Fy,Px,Py\n");
//    appendFile(SD, "/full_set_readings.txt", "Fx,Fy,Px,Py\n");
    Serial.printf("Total space: %lluMB\n", SD.totalBytes() / (1024 * 1024));
    Serial.printf("Used space: %lluMB\n", SD.usedBytes() / (1024 * 1024));

    // Initialize SD Card's writing buffer.
    new_reading_char_ptr = new_reading_char;

    /* Initializing fft arrays. */
    fft_init(true);
    fft_init(false);

    /* Output leds setup*/
    pinMode (led_green_Pin, OUTPUT);
    pinMode (led_red_Pin, OUTPUT);
    digitalWrite (led_green_Pin, LOW);  // turn off the green LED
    digitalWrite (led_red_Pin, LOW);  // turn off the green LED

    /* Timestamps handling init.*/
    //connect to WiFi
     Serial.printf("Connecting to %s ", ssid);
     WiFi.begin(ssid, password);
     while (WiFi.status() != WL_CONNECTED) {
         delay(500);
         Serial.print(".");
     }
     Serial.println(" CONNECTED");

     //init and get the time
     configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);
//     printLocalTime();

    /* Email handling init.*/

    // Set the SMTP Server Email host, port, account and password
    smtpData.setLogin(smtpServer, smtpServerPort, emailSenderAccount, emailSenderPassword);

    // Set the sender name and Email
    smtpData.setSender("ESP32", emailSenderAccount);

    // Set Email priority or importance High, Normal, Low or 1 to 5 (1 is highest)
    smtpData.setPriority("Normal");

    // Set the subject
    smtpData.setSubject(emailSubject);

    // Set the message with HTML format
    smtpData.setMessage("<div style=\"color:#2f4468;\"><h1>Your system is not working correctly. Please call your technician. </h1><p>- Sent from Kresnas 11 ~ Pump Operation Detector</p></div>", true);

    // Add recipients, you can add more than one recipient
    smtpData.addRecipient(emailRecipient);

    /* FREERTOS Tasks Initialization */
    xTaskCreatePinnedToCore(
      FFT_FUN, // Function to implement the task
      "FFT_TASK", // Name of the task
      10000,  // Stack size in words
      NULL,  // Task input parameter
      2,  // Priority of the task
      &FFT_FUN_Task,  // Task handle.
      0); // Core where the task should run

    xTaskCreatePinnedToCore(
      ML_NN, // Function to implement the task
      "ML_NN_TASK", // Name of the task
      10000,  // Stack size in words
      NULL,  // Task input parameter
      3,  // Priority of the task
      &ML_Task,  // Task handle.
      0); // Core where the task should run

    xTaskCreatePinnedToCore(
      MPU6050_READ, // Function to implement the task
      "READ_FROM_MPU_6050SENSOR", // Name of the task
      10000,  // Stack size in words
      NULL,  // Task input parameter
      1,  // Priority of the task
      &MPU6050_Task,  // Task handle.
      0); // Core where the task should run

}

void loop(){

}

/******************************************************************************/
/******************* ML_Neural Network handling functions. ********************/
/******************************************************************************/

void ML_NN( void *parameter ){
    for (;;) /* A Task shall never return or exit. */
    {

        float max = 0;
        float sum = 0;
        float max_ampl_x = 0;
        float max_ampl_y = 0;
        float min = 10000;

        // Calculate median peak freq of x axis vibrations.

        for(int k=0;k<10;k++){
          if(input[k]>=max){
            max = input[k];
          }
          if(input[k]<=min){
            min = input[k];
          }
        }

        for(int k=0;k<10;k++){
          if(input[k]!= max && input[k]!= min){
           sum += input[k];
           max_ampl_x += ampl_x[k];
          }
          //Serial.print(input[k]);
          //Serial.print(",");
          input[k]= 0;
        }
        //Serial.println(min);
        //Serial.println(max);

        // Feeding the Neural Network.
        input_nn[2] = max_ampl_x/8;
        input_nn[0] = sum/8;

        // Calculate median peak freq of y axis vibrations.

        max = 0;
        min = 10000;
        sum = 0;

        for(int k=10;k<20;k++){
          if(input[k]>=max){
            max = input[k];
          }
          if(input[k]<=min){
            min = input[k];
          }
        }

        for(int k=10;k<20;k++){
          if(input[k]!=max && input[k]!=min){
            sum += input[k];
            max_ampl_y += ampl_y[k];
            //Serial.print(input[k]);
            //Serial.print(",");
          }
          input[k]= 0;
        }

        //Serial.println(min);
        //Serial.println(max);
        //Serial.println("");

        // Feeding the Neural Network.
        input_nn[1] = sum/8;
        input_nn[3] = max_ampl_y/8;

        predicted = ml.predict(input_nn);
        //Serial.println(predicted,5);

        if(predicted < 0){
          predicted = 0;
        }else{
          predicted = round(abs(predicted));
        }

        /* Show physical output through the leds*/

        if(output_timer > 2){
            if(predicted == 2){
              digitalWrite (led_green_Pin, LOW);  // turn off the green LED
              digitalWrite (led_red_Pin, HIGH);  // turn on the red LED
              //Send an Email Allert.
              if(email_enable == 1){
                if (!MailClient.sendMail(smtpData)){
                  Serial.println("Error sending Email, " + MailClient.smtpErrorReason());
                  //Clear all data from Email object to free memory
                  smtpData.empty();
                }
                email_enable = 0;
              }
            }
            else if (predicted == 1 && input_nn[2] > 0.5 && input_nn[3] > 0.5){
              digitalWrite (led_green_Pin, HIGH);  // turn on the green LED
              digitalWrite (led_red_Pin, LOW);  // turn offthe red LED
            }
            else{
              digitalWrite (led_green_Pin, LOW);  // turn on the green LED
              digitalWrite (led_red_Pin, LOW);  // turn offthe red LED
            }
        }
        else{
         output_timer++;
        }

        printLocalTime();

        Serial.print(input_nn[0]);
        Serial.print(",");
        new_reading_char_ptr  += sprintf(new_reading_char_ptr, "%f,", input_nn[0]);
        Serial.print(input_nn[1]);
        new_reading_char_ptr  += sprintf(new_reading_char_ptr, "%f,", input_nn[1]);
        Serial.print(",");
        Serial.print(input_nn[2],6);
        Serial.print(",");
        Serial.print(input_nn[3],6);
         Serial.println("");
//        Serial.print(max_ampl_x/8,6);
        new_reading_char_ptr  += sprintf(new_reading_char_ptr, "%f,", max_ampl_x/8);
//        Serial.print(",");
//        Serial.print(max_ampl_y/8,6);
        new_reading_char_ptr  += sprintf(new_reading_char_ptr, "%f\n", max_ampl_y/8);
        Serial.println("");

        Serial.println(predicted,5);
        new_reading_char_ptr  += sprintf(new_reading_char_ptr, "%f\n", predicted);

        predicted = 0.0;
        j = 0;

//       Serial.println(new_reading_char);
        appendFile(SD, "/full_set_readings.txt", new_reading_char);
        new_reading_char_ptr = new_reading_char;
        delay( 10000 ); // wait for 10 seconds
    }
}

/******************************************************************************/
/************************ FFT handling functions. *****************************/
/******************************************************************************/

void FFT_FUN( void *parameter ){
    for (;;) /* A Task shall never return or exit. */
    {
        // i should be 128 since the samples are 128.
        // j is the number of time the task ran and should be [0,9]
//        Serial.println(i);
//        Serial.println(vReal_x[127],6);
//        Serial.println(vReal_y[127],6);
        //FFT.Windowing(vReal_x, samples, FFT_WIN_TYP_HAMMING, FFT_FORWARD);  /* Weigh data */
        FFT.Compute(vReal_x, vImag, samples, FFT_FORWARD); /* Compute FFT */

        //filtering out Low frequencies ~ noise cancelling [0,15Hz]
//        for(int k=0;k<15;k++){
//          vReal_x[k] =0;
//          vImag[k] = 0;
//        }

        FFT.ComplexToMagnitude(vReal_x, vImag, samples); /* Compute magnitudes */
        double x = FFT.MajorPeak(vReal_x, samples, samplingFrequency);
        if(j<10){
          input[j] = x;                                 // Peak Frequency of Ax
//          Serial.println(x);

          ampl_x[j] = vReal_x[(int)x];                  // Amplitude for the max frequency component.(x axis) in Gs
//          ampl_x[j] = (vReal_x[0]/samples)*1000;     // DC/128 -> mid amplitude (x axis) in millimeter/sec^2

        }

        // Reset the arrays for the next measurments.
        fft_init(true);

        //FFT.Windowing(vReal_y, samples, FFT_WIN_TYP_HAMMING, FFT_FORWARD);  /* Weigh data */
        FFT.Compute(vReal_y, vImag, samples, FFT_FORWARD); /* Compute FFT */

        //filtering out Low frequencies ~ noise cancelling [0,15Hz]
//        for(int k=0;k<15;k++){
//          vReal_y[k] =0;
//          vImag[k] = 0;
//        }

        FFT.ComplexToMagnitude(vReal_y, vImag, samples); /* Compute magnitudes */
        double y = FFT.MajorPeak(vReal_y, samples, samplingFrequency);
        if(j<10){
          input[j+10] = y;                             // Peak Frequency of Ay
//
//        Serial.println(y);
//          Serial.println(vReal_y[(int)y]);
//          ampl_y[j] = (vReal_y[0]/samples)*1000;     // DC/128 -> mid amplitude (y axis) in Gs
          ampl_y[j] = vReal_y[(int)y];                 // Amplitude for the max frequency component.(y axis) in Gs

          j++; // Increment the Neural Network's input counter;
        }

        // Reset the arrays for the next measurments.
        fft_init(false);

        delay( 1000 ); // wait for one second // 1028

    }
}

/******************************************************************************/
/******************** MPU6050 sensor handling functions. **********************/
/******************************************************************************/

void MPU6050_READ( void *parameter ){
    for (;;) /* A Task shall never return or exit. */
    {

      //Serial.println("Here");
      Read_RawValue(MPU6050SlaveAddress, MPU6050_REGISTER_ACCEL_XOUT_H);

      /* Divide each with their sensitivity scale factor */

      Ax = abs((double)AccelX/AccelScaleFactor);

      if(i<samples) {
        vReal_x[i] = Ax;
      }

      Ay = abs((double)AccelY/AccelScaleFactor);
      if(i<samples) {
        vReal_y[i] = Ay;
      }
//
//      Serial.print(Ax,6);
//      Serial.print(",");
//      Serial.print(Ay,6);
//      Serial.println("");

//      Az = abs((double)AccelZ/AccelScaleFactor);
//      if(i<samples) {
//        vReal_z[i] = Az;
//      }

      i++;
      delayMicroseconds(900); // 256 Hz -> 760 900 // 128 Hz -> 2750 2600
    }
}

void I2C_Write(uint8_t deviceAddress, uint8_t regAddress, uint8_t data){
  Wire.beginTransmission(deviceAddress);
  Wire.write(regAddress);
  Wire.write(data);
  Wire.endTransmission();
}

/* Reading only the 3 registers containing acceleration reading */

void Read_RawValue(uint8_t deviceAddress, uint8_t regAddress){
  Wire.beginTransmission(deviceAddress);
  Wire.write(regAddress);
  Wire.endTransmission();
  Wire.requestFrom(deviceAddress, (uint8_t)6); // Reading only Accel readings.
  AccelX = (((int16_t)Wire.read()<<8) | Wire.read()) + Ax_offset; // Ax_MSB + Ax_LSB
  AccelY = (((int16_t)Wire.read()<<8) | Wire.read()) + Ay_offset; // Ay_MSB + Ay_LSB
//  AccelZ = (((int16_t)Wire.read()<<8) | Wire.read()); // Az_MSB + Az_LSB
}

/* MPU6050 configuration */

void MPU6050_Init(){
  delay(150);

  // MPU6050 reset the registers
  I2C_Write(MPU6050SlaveAddress, MPU6050_REGISTER_PWR_MGMT_1, 0x40); // 0x01 & 0x08

  delay(150);
  // specifies the divider from the gyroscope output rate used to generate the Sample Rate
  I2C_Write(MPU6050SlaveAddress, MPU6050_REGISTER_SMPLRT_DIV, 0x7F); // 0x07

  // selecting internal clock source PLL with X axis gyroscope reference and disabling temperature readings.
  I2C_Write(MPU6050SlaveAddress, MPU6050_REGISTER_PWR_MGMT_1, 0x08); // 0x01 & 0x08

  // 1 - This register allows the user to configure the frequency of wake-ups in Accelerometer Only Low Power Mode.
  // 2 - This register also allows the user to put individual axes of the accelerometer and gyroscope into standby mode/
  I2C_Write(MPU6050SlaveAddress, MPU6050_REGISTER_PWR_MGMT_2, 0x07);

  // configures the bandwidth of the low pass filter at 260Hz -> DELAY 0ms
  I2C_Write(MPU6050SlaveAddress, MPU6050_REGISTER_CONFIG, 0x00);

  // set +/- 2g full scale range for better sensitivity
  I2C_Write(MPU6050SlaveAddress, MPU6050_REGISTER_ACCEL_CONFIG, 0x00);

  // determines which sensor measurements are loaded into the FIFO buffer -> Bit3 is Acce Data
  I2C_Write(MPU6050SlaveAddress, MPU6050_REGISTER_FIFO_EN, 0x08);

  // bit0 enables the Data Ready interrupt,which occurs each time a write operation to all of the sensor's registers has been completed
  I2C_Write(MPU6050SlaveAddress, MPU6050_REGISTER_INT_ENABLE, 0x00);

  // for future use if we want to reset the A/D signal paths
  I2C_Write(MPU6050SlaveAddress, MPU6050_REGISTER_SIGNAL_PATH_RESET, 0x00);

  // this register allows the user to enable/disable the FIFO buffer,I2C MasterMode,primary I2C interface.
  I2C_Write(MPU6050SlaveAddress, MPU6050_REGISTER_USER_CTRL, 0x40); //Enable FIFO Operations. //0x00
}

void fft_init(bool xy){
  /* FFT Preparation */
    // initialize arrays for Fx
    if(xy){
      for(i=0;i<samples;i++){
        vReal_x[i] = 0.0;
        vImag[i] = 0.0;
      }
    // initialize arrays for Fx
    }else{
      for(i=0;i<samples;i++){
        vReal_y[i] = 0.0;
        vImag[i] = 0.0;
      }
    }

    i = 0;
}

void print_freq_magn(){
  /* FFT Preparation */
    for(i=0;i<samples;i++){
       Serial.println(vReal_x[i],6);
    }
}

/*********************** SD Card functions **************************/

void writeFile(fs::FS &fs, const char * path, const char * message){
    Serial.printf("Writing file: %s\n", path);

    File file = fs.open(path, FILE_WRITE);
    if(!file){
        Serial.println("Failed to open file for writing");
        return;
    }
    if(file.print(message)){
        //Serial.println("File written");
    } else {
        Serial.println("Write failed");
    }
    file.close();
}

void appendFile(fs::FS &fs, const char * path, const char * message){
    //Serial.printf("Appending to file: %s\n", path);

    File file = fs.open(path, FILE_APPEND);
    if(!file){
        Serial.println("Failed to open file for appending");
        return;
    }
    if(file.print(message)){
        //Serial.println("Message appended");
    } else {
        Serial.println("Append failed");
    }
    file.close();
}

/* Time stamps handling.*/
void printLocalTime()
{
  struct tm timeinfo;
  char time_array[35];
  if(!getLocalTime(&timeinfo)){
    Serial.print("Failed to obtain time");
    return;
  }
//  Serial.println(&timeinfo, "%A, %B %d %Y %H:%M:%S");
//  new_reading_char_ptr  += sprintf(new_reading_char_ptr,"%A, %B %d %Y %H:%M:%S \n",timeinfo);
    strftime(time_array, 35, "%A, %B %d %Y %H:%M:%S \n",&timeinfo);
    new_reading_char_ptr += sprintf(new_reading_char_ptr,"%s",time_array);

//    memcpy(new_reading_char_ptr,time_array,35);
//    new_reading_char_ptr  += sprintf(new_reading_char_ptr,"%A, %B %d %Y %H:%M:%S \n",timeinfo);
//  Serial.println(new_reading_char );
}
