#include "utilities.hpp"

int64_t time_in_ms()
{
    auto now = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    return static_cast<int64_t>(ms);
}

static void log_debug(const char *format, ...)
{
    static FILE *debug_file = NULL;
    va_list args;

    // Open file on first use
    if (!debug_file)
    {
        debug_file = fopen("debug.log", "w");
        if (!debug_file)
            return;
    }

    va_start(args, format);
    vfprintf(debug_file, format, args);
    va_end(args);

    // Flush after each write to ensure we get logs even if program crashes
    fflush(debug_file);
}

static void read_weights_from_file(FILE *file, float32_t *dest, size_t n_elements)
{
    float32_t *temp;
    size_t i, chunk_size, current_chunk, remaining;
    const size_t MAX_CHUNK = 16384; // Read in 16KB chunks

    if (sizeof(float32_t) == sizeof(float32_t))
    {
        // Direct read for 32-bit platforms
        remaining = n_elements;
        while (remaining > 0)
        {
            chunk_size = (remaining < MAX_CHUNK) ? remaining : MAX_CHUNK;
            if (fread(dest, sizeof(float), chunk_size, file) != chunk_size)
            {
                log_debug("Failed to read chunk of %ld elements\n", (long)chunk_size);
                exit(EXIT_FAILURE);
            }
            dest += chunk_size;
            remaining -= chunk_size;
        }
    }
    else
    {
        // Read into temp buffer and convert for other platforms
        temp = (float *)malloc(MAX_CHUNK * sizeof(float));
        if (!temp)
        {
            log_debug("Failed to allocate temp buffer for %ld elements\n", (long)MAX_CHUNK);
            exit(EXIT_FAILURE);
        }

        remaining = n_elements;
        while (remaining > 0)
        {
            current_chunk = (remaining < MAX_CHUNK) ? remaining : MAX_CHUNK;

            if (fread(temp, sizeof(float), current_chunk, file) != current_chunk)
            {
                log_debug("Failed to read chunk of %ld elements (remaining: %ld)\n",
                          (long)current_chunk, (long)remaining);
                free(temp);
                exit(EXIT_FAILURE);
            }

            for (i = 0; i < current_chunk; i++)
            {
                dest[i] = (float)temp[i];
            }

            dest += current_chunk;
            remaining -= current_chunk;
        }

        free(temp);
    }
}