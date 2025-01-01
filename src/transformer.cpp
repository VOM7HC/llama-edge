#include "transformer.hpp"
#include "utilities.hpp"

Config::Config()
{
}

Config::~Config()
{
}

TransformerWeights::TransformerWeights()
{
}

TransformerWeights::~TransformerWeights()
{
}

Transformer::Transformer()
{
}

Transformer::~Transformer()
{
}

RunState::RunState()
{
}

RunState::~RunState()
{
}

void malloc_run_state(RunState *s, Config *p)
{
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    printf("Allocating RunState buffers: dim=%d, hidden_dim=%d, kv_dim=%d\n",
           p->dim, p->hidden_dim, kv_dim);

    s->x = (float *)calloc(p->dim, sizeof(float));
    s->xb = (float *)calloc(p->dim, sizeof(float));
    s->xb2 = (float *)calloc(p->dim, sizeof(float));
    s->hb = (float *)calloc(p->hidden_dim, sizeof(float));
    s->hb2 = (float *)calloc(p->hidden_dim, sizeof(float));
    s->q = (float *)calloc(p->dim, sizeof(float));
    s->key_cache = (float *)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = (float *)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->att = (float *)calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = (float *)calloc(p->vocab_size, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->key_cache || !s->value_cache || !s->att || !s->logits)
    {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
    printf("All RunState buffers allocated successfully\n");
}

void free_run_state(RunState *s)
{
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

void read_checkpoint(int8_t *checkpoint, Config *config, TransformerWeights *weights,
                     int32_t *fd, float32_t **data, int64_t *file_size)
{
    FILE *file;
    int shared_weights;
    size_t offset;
    int head_size;
    size_t layer_size;
    size_t embedding_size;
    const size_t chunk_size = 64 * 1024; // 64KB chunks
    size_t remaining, current_chunk;
    char *ptr;
    long current_pos;
    size_t bytes_read;
    size_t read_size;
    size_t rms_size;
    size_t total_allocated, alloc_size;
    const size_t MAX_SINGLE_ALLOC = 1024 * 1024; // 1MB max single allocation

    file = fopen((const char*)checkpoint, "rb");
    if (!file)
    {
        fprintf(stderr, "Couldn't open file %s\n", checkpoint);
        exit(EXIT_FAILURE);
    }

    // read in the config header
    if (fread(config, sizeof(Config), 1, file) != 1)
    {
        exit(EXIT_FAILURE);
    }

    shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    head_size = config->dim / config->n_heads;

    fseek(file, 0, SEEK_END);
    *file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    log_debug((const int8_t*)"Reading checkpoint: file_size=%ld bytes (%.2f MB)\n",
              *file_size, (float)*file_size / (1024 * 1024));
    log_debug((const int8_t*)"Config: vocab_size=%d, dim=%d, n_layers=%d\n",
              config->vocab_size, config->dim, config->n_layers);

    // Skip the config header for subsequent reads
    offset = sizeof(Config);

    // token embedding table
    embedding_size = config->vocab_size * config->dim;
    log_debug((const int8_t*)"Allocating token_embedding_table: %ld elements (%ld bytes)\n",
              (long)embedding_size, (long)(embedding_size * sizeof(float)));

    weights->token_embedding_table = (float *)malloc(embedding_size * sizeof(float));
    if (!weights->token_embedding_table)
    {
        log_debug((const int8_t*)"Failed to allocate token_embedding_table\n");
        exit(EXIT_FAILURE);
    }
    log_debug((const int8_t*)"Allocated token_embedding_table at %p\n", (void *)weights->token_embedding_table);

    // Read in chunks
    remaining = embedding_size * sizeof(float);
    ptr = (char *)weights->token_embedding_table;
    fseek(file, offset, SEEK_SET);

    while (remaining > 0)
    {
        current_pos = ftell(file);
        current_chunk = remaining < chunk_size ? remaining : chunk_size;

        bytes_read = fread(ptr, 1, current_chunk, file);
        if (bytes_read != current_chunk)
        {
            log_debug((const int8_t*)"Failed to read chunk: expected %ld bytes, got %ld bytes\n",
                      (long)current_chunk, (long)bytes_read);
            log_debug((const int8_t*)"File position: %ld, file size: %ld\n",
                      ftell(file), *file_size);
            exit(EXIT_FAILURE);
        }

        ptr += current_chunk;
        remaining -= current_chunk;
    }

    offset += embedding_size * sizeof(float);
    log_debug((const int8_t*)"Successfully read token_embedding_table in chunks\n");

    // rms attention weights
    rms_size = config->n_layers * config->dim;
    weights->rms_att_weight = (float *)malloc(rms_size * sizeof(float));
    if (!weights->rms_att_weight)
    {
        log_debug((const int8_t*)"Failed to allocate rms_att_weight\n");
        exit(EXIT_FAILURE);
    }
    fseek(file, offset, SEEK_SET);
    read_weights_from_file(file, weights->rms_att_weight, rms_size);
    offset += rms_size * sizeof(float);

    // wq, wk, wv weights
    layer_size = config->dim * config->n_heads * head_size;
    weights->wq = (float *)malloc(config->n_layers * layer_size * sizeof(float));
    log_debug((const int8_t*)"Allocated wq at %p\n", (void *)weights->wq);
    fseek(file, offset, SEEK_SET);
    read_weights_from_file(file, weights->wq, config->n_layers * layer_size);
    offset += config->n_layers * layer_size * sizeof(float);

    layer_size = config->dim * config->n_kv_heads * head_size;
    weights->wk = (float *)malloc(config->n_layers * layer_size * sizeof(float));
    log_debug((const int8_t*)"Allocated wk at %p\n", (void *)weights->wk);
    fseek(file, offset, SEEK_SET);
    read_weights_from_file(file, weights->wk, config->n_layers * layer_size);
    offset += config->n_layers * layer_size * sizeof(float);

    weights->wv = (float *)malloc(config->n_layers * layer_size * sizeof(float));
    log_debug((const int8_t*)"Allocated wv at %p\n", (void *)weights->wv);
    fseek(file, offset, SEEK_SET);
    read_weights_from_file(file, weights->wv, config->n_layers * layer_size);
    offset += config->n_layers * layer_size * sizeof(float);

    // wo weights
    layer_size = config->n_heads * head_size * config->dim;
    weights->wo = (float *)malloc(config->n_layers * layer_size * sizeof(float));
    log_debug((const int8_t*)"Allocated wo at %p\n", (void *)weights->wo);
    fseek(file, offset, SEEK_SET);
    read_weights_from_file(file, weights->wo, config->n_layers * layer_size);
    offset += config->n_layers * layer_size * sizeof(float);

    // Remaining weights...
    weights->rms_ffn_weight = (float *)malloc(config->n_layers * config->dim * sizeof(float));
    fseek(file, offset, SEEK_SET);
    read_weights_from_file(file, weights->rms_ffn_weight, config->n_layers * config->dim);
    offset += config->n_layers * config->dim * sizeof(float);

    // w1, w2, w3 weights
    layer_size = config->dim * config->hidden_dim;
    log_debug((const int8_t*)"Allocating w1: layer_size=%ld, total bytes=%ld\n",
              (long)layer_size, (long)(config->n_layers * layer_size * sizeof(float)));

    // Add memory tracking
    total_allocated = 0;

    // When allocating large buffers:
    alloc_size = config->n_layers * layer_size * sizeof(float);
    if (alloc_size > MAX_SINGLE_ALLOC)
    {
        log_debug((const int8_t*)"Warning: Large allocation of %ld bytes requested\n", (long)alloc_size);
    }

    weights->w1 = (float *)malloc(alloc_size);
    if (!weights->w1)
    {
        log_debug((const int8_t*)"Failed to allocate w1 (%ld bytes)\n", (long)alloc_size);
        // Clean up previous allocations
        // Add cleanup code here
        exit(EXIT_FAILURE);
    }
    total_allocated += alloc_size;
    log_debug((const int8_t*)"Total memory allocated: %ld bytes\n", (long)total_allocated);

    fseek(file, offset, SEEK_SET);
    read_weights_from_file(file, weights->w1, config->n_layers * layer_size);
    offset += config->n_layers * layer_size * sizeof(float);

    layer_size = config->dim * config->hidden_dim;
    weights->w2 = (float *)malloc(config->n_layers * layer_size * sizeof(float));
    weights->w3 = (float *)malloc(config->n_layers * layer_size * sizeof(float));

    fseek(file, offset, SEEK_SET);
    read_weights_from_file(file, weights->w2, config->n_layers * layer_size);
    offset += config->n_layers * layer_size * sizeof(float);

    fseek(file, offset, SEEK_SET);
    read_weights_from_file(file, weights->w3, config->n_layers * layer_size);
    offset += config->n_layers * layer_size * sizeof(float);

    // final rms norm
    weights->rms_final_weight = (float *)malloc(config->dim * sizeof(float));
    fseek(file, offset, SEEK_SET);
    read_weights_from_file(file, weights->rms_final_weight, config->dim);
    offset += config->dim * sizeof(float);

    // Skip freq_cis_real and freq_cis_imag
    offset += config->seq_len * head_size * sizeof(float);

    // classifier weights
    if (!shared_weights)
    {
        weights->wcls = (float *)malloc(config->vocab_size * config->dim * sizeof(float));
        fseek(file, offset, SEEK_SET);
        read_weights_from_file(file, weights->wcls, config->vocab_size * config->dim);
    }
    else
    {
        weights->wcls = weights->token_embedding_table;
    }

    log_debug((const int8_t*)"Checkpoint loaded successfully\n");
    fclose(file);
}

void build_transformer(Transformer *t, int8_t *checkpoint_path)
{
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer *t)
{
    // Replace munmap with free
    if (t->data)
    {
        free(t->data);
    }
    // No need to close fd since we're not using mmap
    free_run_state(&t->state);
}