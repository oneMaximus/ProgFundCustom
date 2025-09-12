#include <stdio.h>
#include "pico/stdlib.h"
#include "hardware/uart.h"
#include "hardware/gpio.h"
#include "functions.h"

#define UART_ID uart1
#define BAUD_RATE 9600
#define UART_TX_PIN 8   // GP8 = TX
#define UART_RX_PIN 9   // GP9 = RX
#define BUTTON_PIN 22   // GP22 = button

int main() {
    stdio_usb_init();

    uart_init(UART_ID, BAUD_RATE);
    gpio_set_function(UART_TX_PIN, GPIO_FUNC_UART);
    gpio_set_function(UART_RX_PIN, GPIO_FUNC_UART);

    gpio_init(BUTTON_PIN);
    gpio_set_dir(BUTTON_PIN, GPIO_IN);
    gpio_pull_up(BUTTON_PIN);

    uint8_t letter = 'A';

    while (true) {
        bool button_pressed = (gpio_get(BUTTON_PIN) == 0);
        char tx_char = get_transmit_char(button_pressed, &letter);
        uart_putc(UART_ID, tx_char);

        while (uart_is_readable(UART_ID)) {
            char rx_char = uart_getc(UART_ID);
            char processed = process_received_char(rx_char);
            if (processed != 0) {
                printf("%c\n", processed);
            }
        }

        sleep_ms(1000);
    }
}
