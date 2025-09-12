#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <stdint.h>
#include <stdbool.h>

// Determines which character to transmit based on button state.
// `button_pressed` = true if button is pressed.
// `current_char` = pointer to current alphabet character (A-Z), updated if button is pressed.
char get_transmit_char(bool button_pressed, uint8_t *current_char);

// Processes a received character and returns the transformed character.
// Uppercase letters -> lowercase, '1' -> '2', others ignored.
char process_received_char(char received);

#endif // FUNCTIONS_H
