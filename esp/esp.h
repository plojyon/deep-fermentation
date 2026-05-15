#pragma once

#define STATUS_LED 2

/*********/
/* Wi-Fi */
/*********/
struct WifiNetwork {
	const char* ssid;
	const char* passwd;
};
#define WIFI_CHECK_INTERVAL_SECONDS 10
#define CONNECT_TO_OPEN_NETWORKS false

/**************/
/* Microphone */
/**************/
// Connections to INMP441 I2S microphone
#define I2S_WS 25
#define I2S_SD 33
#define I2S_SCK 32
#define I2S_PORT I2S_NUM_0

// Sampling settings
#define SAMPLE_RATE 44100
#define SAMPLE_BUFFER_SIZE 4410
