#include <math.h>
#include <stdint.h>
#include <string.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <driver/i2s.h>
#include <WiFiClientSecure.h>
#include <PubSubClient.h>
#include <arduinoFFT.h>

#include <ArduinoWebsockets.h>
using namespace websockets;

#include "esp.h"
#include "detector.h"
#include "wifi_secrets.h"
#include "mqtt_secrets.h"

#include "detector_params.h"

/*****************************************************************************/
// Microphone //

void i2s_install() {
	// Set up I2S Processor configuration
	i2s_config_t i2s_config = {
		.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
		.sample_rate = SAMPLE_RATE,
		.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
		.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
		.communication_format = I2S_COMM_FORMAT_I2S,
		.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
		.dma_buf_count = 4,
		.dma_buf_len = 1024,
		.use_apll = false,
		.tx_desc_auto_clear = false,
		.fixed_mclk = 0
	};

	i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
}

void i2s_setpin() {
	// Set I2S pin configuration
	const i2s_pin_config_t pin_config = {
		.bck_io_num = I2S_SCK,
		.ws_io_num = I2S_WS,
		.data_out_num = I2S_PIN_NO_CHANGE,
		.data_in_num = I2S_SD
	};

	i2s_set_pin(I2S_PORT, &pin_config);
}


/*****************************************************************************/
// Wi-Fi //

bool wifi_connected = false;

void initWiFiBlocking(const char* ssid, const char* password) {
	delay(1000);
	Serial.print("Connecting to WiFi (");
	Serial.print(ssid);
	Serial.println(")");
	if (password == NULL) {
		WiFi.begin(ssid);
	} else {
		WiFi.begin(ssid, password);
	}
	WiFi.printDiag(Serial);
	bool connected = true;
	for (int i = 0; WiFi.status() != WL_CONNECTED; i++) {
		Serial.print('.');
		Serial.print(WiFi.status());
		Serial.print('.');
		digitalWrite(STATUS_LED, LOW);
		delay(100);
		digitalWrite(STATUS_LED, HIGH);
		delay(100);
		digitalWrite(STATUS_LED, LOW);
		delay(100);
		digitalWrite(STATUS_LED, HIGH);
		delay(1000);
		if (i > 30) {
			connected = false;
			break;
		}
	}
	if (connected) {
		wifi_connected = true;
		Serial.print("Connected with IP ");
		Serial.println(WiFi.localIP());
	} else {
		wifi_connected = false;
		Serial.println("Unable to connect");
	}
}

const char* get_passwd(const char* ssid) {
	const size_t n_networks = sizeof(KNOWN_NETWORKS) / sizeof(KNOWN_NETWORKS[0]);
	for (int i = 0; i < n_networks; i++) {
		if (strcmp(KNOWN_NETWORKS[i].ssid, ssid) == 0) {
			return KNOWN_NETWORKS[i].passwd;
		}
	}
	return NULL;
}

void connect_to_wifi() {
	WiFi.disconnect(true);
	delay(100);
	const long T_start = micros();
	int n = WiFi.scanNetworks();
	const long T_end = micros();
	Serial.printf("Done in %dms\n", (int)((T_end - T_start) / 1000));
	if (n == 0) {
		Serial.println("no networks found");
		delay(5000);
	} else {
		Serial.print(n);
		Serial.println(" networks found");

		const char* ssid = NULL;
		const char* passwd = NULL;

		// Hopefully they are sorted by RSSI
		for (int i = 0; i < n; ++i) {
			String ssid_str = WiFi.SSID(i);
			const char* current_ssid = ssid_str.c_str();
			Serial.print("Checking network ");
			Serial.print(current_ssid);
			Serial.print("\n");
			const char* current_passwd = get_passwd(current_ssid);
			if (current_passwd != NULL) {
				Serial.println("Network is known!");
				Serial.print(current_ssid);
				Serial.print(" - ");
				Serial.print(current_passwd);
				Serial.println(" baje");
				ssid = current_ssid;
				passwd = current_passwd;
				break;
			} else if (WiFi.encryptionType(i) == WIFI_AUTH_OPEN && CONNECT_TO_OPEN_NETWORKS && ssid == NULL) {
				Serial.print("Network is open! No better networks available yet, picking this one.");
				ssid = current_ssid;
			}
		}
		if (ssid != NULL) {
			initWiFiBlocking(ssid, passwd);
		}
		else {
			delay(500);
		}
	}
}

const int wifi_check_delay = WIFI_CHECK_INTERVAL_SECONDS * 1000;
int wifi_last_check = 0;

void wifi_loop() {
	if (millis() - wifi_last_check > wifi_check_delay) {
		wifi_last_check = millis();

		if (WiFi.status() != WL_CONNECTED) {
			wifi_connected = false;
		} else {
			wifi_connected = true;
		}
	}
}

/*****************************************************************************/
// Websockets //

const char* WS_ENDPOINT = "ws://192.168.1.115:8080/ws";
WebsocketsClient ws_client;
bool ws_connected = false;
bool is_connecting = false;

void init_websocket() {
	if(!is_connecting) {
		is_connecting = true;
		bool connected = ws_client.connect(WS_ENDPOINT);
		is_connecting = false;
		if(connected) {
			Serial.println("WS Connected!");
		} else {
			Serial.println("WS Not Connected!");
			delay(1000);
		}
		ws_connected = connected;
	}
}
void onEventsCallback(WebsocketsEvent event, String data) {
	if(event == WebsocketsEvent::ConnectionOpened) {
		Serial.println("Connnection Opened");
	} else if(event == WebsocketsEvent::ConnectionClosed) {
		Serial.println("Connnection Closed");
		ws_connected = false;
		ws_client.close();
		init_websocket();
	} else if(event == WebsocketsEvent::GotPing) {
		Serial.println("Got a Ping!");
	} else if(event == WebsocketsEvent::GotPong) {
		Serial.println("Got a Pong!");
	}
}

/*****************************************************************************/
// Detector //

// Store 4 bytes for each 32 bit sample
uint8_t buffer32[SAMPLE_BUFFER_SIZE * 4];
int16_t buffer16[SAMPLE_BUFFER_SIZE];

void detector_setup() {
	Serial.println("Hello bubbles!");
}

void detector_loop() {
	// delay(1000000000);
	// Serial.println("Detecting bubbles");

	// Get I2S data and place in data buffer
	size_t bytes_read = 0;

	esp_err_t result = i2s_read(I2S_NUM_0, buffer32, sizeof(int32_t) * SAMPLE_BUFFER_SIZE, &bytes_read, portMAX_DELAY);
	int samples_read = bytes_read / sizeof(int32_t);
	for (int i = 0; i < samples_read; i++) {
		uint8_t mid = buffer32[i * 4 + 2];
		uint8_t msb = buffer32[i * 4 + 3];
		uint16_t raw = (((uint32_t)msb) << 8) + ((uint32_t)mid);
		memcpy(&buffer16[i], &raw, sizeof(raw)); // Copy so sign bits aren't interfered with somehow.
	}

	if (result == ESP_OK) {
		// Serial.println(buffer16[0]);
		ws_client.sendBinary((const char*)buffer16, samples_read * sizeof(int16_t));
	}
	else {
		Serial.print("Result of reading is ");
		Serial.println(result);
	}
}


/*****************************************************************************/
// Setup and Loop //

void setup() {
	// station mode
	WiFi.mode(WIFI_STA);
	WiFi.disconnect();
	delay(100);
	Serial.begin(115200);

	// Set up I2S
	i2s_install();
	i2s_setpin();
	i2s_start(I2S_PORT);

	detector_setup();

	Serial.println("Setup done!");

	pinMode(STATUS_LED, OUTPUT);
}

void loop() {
	if (wifi_connected) {
		detector_loop();
	} else {
		connect_to_wifi();
		init_websocket();
	}
	wifi_loop();
}

