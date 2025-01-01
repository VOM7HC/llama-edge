#include "pch.hpp"

#if defined(__linux__)
typedef float float32_t;
#endif

#define private public

int64_t time_in_ms();
static void log_debug(const int8_t* format, ...);
static void read_weights_from_file(FILE* file, float32_t* dest, size_t n_elements);