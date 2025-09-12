#include "functions.h"

char get_transmit_char(bool button_pressed, uint8_t *current_char) {
    if (button_pressed) {
        char out = *current_char;
        (*current_char)++;
        if (*current_char > 'Z') *current_char = 'A';
        return out;
    } else {
        return '1';
    }
}

char process_received_char(char received) {
    if (received >= 'A' && received <= 'Z') return received + 32; // uppercase -> lowercase
    if (received == '1') return '2';                              // 1 -> 2
    return 0;                                                     // ignore
}
