// Transformer model

#include "pch.hpp"
#include "utilities.hpp"

#pragma once

class Config
{
public:
    Config();
    ~Config();

private:
    int32_t dim;        // transformer dimension
    int32_t hidden_dim; // for ffn layers
    int32_t n_layers;   // number of layers
    int32_t n_heads;    // number of query heads
    int32_t n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int32_t vocab_size; // vocabulary size, usually 256 (byte-level)
    int32_t seq_len;    // max sequence length
};

class TransformerWeights
{
public:
    TransformerWeights();
    ~TransformerWeights();

private:
    // token embedding table
    float32_t *token_embedding_table; // (vocab_size, dim)

    // weights for rmsnorms
    float32_t *rms_att_weight; // (layer, dim) rmsnorm weights
    float32_t *rms_ffn_weight; // (layer, dim)

    // weights for matmuls. note dim == n_heads * head_size
    float32_t *wq; // (layer, dim, n_heads * head_size)
    float32_t *wk; // (layer, dim, n_kv_heads * head_size)
    float32_t *wv; // (layer, dim, n_kv_heads * head_size)
    float32_t *wo; // (layer, n_heads * head_size, dim)

    // weights for ffn
    float32_t *w1; // (layer, hidden_dim, dim)
    float32_t *w2; // (layer, dim, hidden_dim)
    float32_t *w3; // (layer, hidden_dim, dim)

    // final rmsnorm
    float32_t *rms_final_weight; // (dim,)

    // (optional) classifier weights for the logits, on the last layer
    float32_t *wcls;
};

class RunState
{
public:
    RunState();
    ~RunState();

private:
    // current wave of activations
    float32_t *x;      // activation at current time stamp (dim,)
    float32_t *xb;     // same, but inside a residual branch (dim,)
    float32_t *xb2;    // an additional buffer just for convenience (dim,)
    float32_t *hb;     // buffer for hidden dimension in the ffn (hidden_dim,)
    float32_t *hb2;    // buffer for hidden dimension in the ffn (hidden_dim,)
    float32_t *q;      // query (dim,)
    float32_t *k;      // key (dim,)
    float32_t *v;      // value (dim,)
    float32_t *att;    // buffer for scores/attention values (n_heads, seq_len)
    float32_t *logits; // output logits

    // kv cache
    float32_t *key_cache;   // (layer, seq_len, dim)
    float32_t *value_cache; // (layer, seq_len, dim)
};

class Transformer
{
public:
    Transformer();
    ~Transformer();

private:
    Config config;              // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state;             // buffers for the "wave" of activations in the forward pass

    // some more state needed to properly clean up the memory mapping (sigh)
    int32_t fd;        // file descriptor for memory mapping
    float32_t *data;   // memory mapped data pointer
    int64_t file_size; // size of the checkpoint file in bytes
};

void malloc_run_state(RunState* s, Config* p);
void free_run_state(RunState* s);
void read_checkpoint(int8_t* checkpoint, Config* config, TransformerWeights* weights,
                     int32_t* fd, float32_t** data, int64_t* file_size);
void build_transformer(Transformer *t, int8_t* checkpoint_path);
void free_transformer(Transformer* t);
