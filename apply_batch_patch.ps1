$ErrorActionPreference = "Stop"

$block = @'
#define PREFILL_BATCH 32

typedef struct {
    GGUFFile *g; const TensorInfo *t; const float *input; float *output;
    uint64_t blocks, first, last; size_t block_bytes, block_elements;
    size_t cols, rows, batch;
    volatile LONG failed;
} BatchMatvecJob;

/* input:  batch vectors, token-major, each `cols` floats
   output: batch vectors, token-major, each `rows` floats */
static void batch_matvec_rows(BatchMatvecJob *job)
{
    uint8_t block[Q6K_BYTES];
    float weights[QK_K];
    float sums[PREFILL_BATCH];
    const size_t n = job->batch;
    for (uint64_t row = job->first; row < job->last; ++row) {
        for (size_t b = 0; b < n; ++b) sums[b] = 0.0f;
        for (uint64_t blk = 0; blk < job->blocks; ++blk) {
            uint64_t absolute = (uint64_t)job->g->data_base + job->t->offset +
                (row * job->blocks + blk) * job->block_bytes;
            if (absolute > g_ram_model.size ||
                job->block_bytes > g_ram_model.size - (size_t)absolute) {
                InterlockedExchange(&job->failed, 1);
                break;
            }
            memcpy(block, g_ram_model.data + (size_t)absolute, job->block_bytes);
            dequant_block(job->t->type, block, weights);   /* once per batch */
            const size_t off = (size_t)blk * job->block_elements;
            const size_t be = job->block_elements;
            size_t b = 0;
            /* four tokens at a time: four independent accumulators */
            for (; b + 4 <= n; b += 4) {
                const float *x0 = job->input + (b + 0) * job->cols + off;
                const float *x1 = job->input + (b + 1) * job->cols + off;
                const float *x2 = job->input + (b + 2) * job->cols + off;
                const float *x3 = job->input + (b + 3) * job->cols + off;
                float s0 = sums[b], s1 = sums[b + 1], s2 = sums[b + 2], s3 = sums[b + 3];
                for (size_t i = 0; i < be; ++i) {
                    float wv = weights[i];
                    s0 += wv * x0[i]; s1 += wv * x1[i];
                    s2 += wv * x2[i]; s3 += wv * x3[i];
                }
                sums[b] = s0; sums[b + 1] = s1; sums[b + 2] = s2; sums[b + 3] = s3;
            }
            for (; b < n; ++b) {
                const float *x = job->input + b * job->cols + off;
                float s = sums[b];
                for (size_t i = 0; i < be; ++i) s += weights[i] * x[i];
                sums[b] = s;
            }
        }
        for (size_t b = 0; b < n; ++b) job->output[b * job->rows + row] = sums[b];
    }
}

static VOID CALLBACK batch_matvec_cb(PTP_CALLBACK_INSTANCE inst, PVOID ctx, PTP_WORK work)
{ (void)inst; (void)work; batch_matvec_rows((BatchMatvecJob *)ctx); }

static int tensor_matvec_cpu_batch(GGUFFile *g, const TensorInfo *t,
                                   const float *input, float *output, size_t batch)
{
    uint64_t cols = t->dims[0], rows = t->dims[1];
    size_t bytes = quant_block_bytes(t->type), be = quant_block_elements(t->type);
    if (!bytes || !be || !cols || cols % be || !rows || !batch || batch > PREFILL_BATCH) return 0;
    SYSTEM_INFO si; GetSystemInfo(&si);
    DWORD workers = si.dwNumberOfProcessors; if (!workers) workers = 1;
    if (workers > 16) workers = 16;
    if (workers > rows) workers = (DWORD)rows;
    BatchMatvecJob jobs[16];
    for (DWORD i = 0; i < workers; ++i) {
        jobs[i] = (BatchMatvecJob){g, t, input, output, cols / be,
            rows * i / workers, rows * (i + 1) / workers,
            bytes, be, (size_t)cols, (size_t)rows, batch, 0};
    }
    PTP_WORK pool_work[16];
    for (DWORD pi = 1; pi < workers; ++pi) {
        pool_work[pi] = CreateThreadpoolWork(batch_matvec_cb, &jobs[pi], NULL);
        if (pool_work[pi]) SubmitThreadpoolWork(pool_work[pi]);
        else batch_matvec_rows(&jobs[pi]);
    }
    batch_matvec_rows(&jobs[0]);
    for (DWORD pi = 1; pi < workers; ++pi)
        if (pool_work[pi]) { WaitForThreadpoolWorkCallbacks(pool_work[pi], FALSE); CloseThreadpoolWork(pool_work[pi]); }
    for (DWORD i = 0; i < workers; ++i) if (jobs[i].failed) return 0;
    return 1;
}

/* look up a tensor by name, check its shape, run the batched matvec */
static int batch_mv(GGUFFile *g, const char *name, const float *in, float *out,
                    size_t n, uint64_t cols, uint64_t rows)
{
    TensorInfo t;
    if (!find_tensor(g, name, &t) || t.n_dims != 2 ||
        t.dims[0] != cols || t.dims[1] != rows) return 0;
    return tensor_matvec_cpu_batch(g, &t, in, out, n);
}

typedef struct { float *x, *proj, *q, *k, *v, *att, *gate, *up, *ffn; } PrefillWork;

static void prefill_work_free(PrefillWork *p)
{
    free(p->x); free(p->proj); free(p->q); free(p->k); free(p->v);
    free(p->att); free(p->gate); free(p->up); free(p->ffn);
    memset(p, 0, sizeof(*p));
}

static int prefill_work_alloc(PrefillWork *p)
{
    const size_t B = PREFILL_BATCH;
    memset(p, 0, sizeof(*p));
    p->x    = calloc(B * HIDDEN_SIZE, sizeof(float));
    p->proj = calloc(B * HIDDEN_SIZE, sizeof(float));
    p->q    = calloc(B * Q_DIM,       sizeof(float));
    p->k    = calloc(B * KV_DIM,      sizeof(float));
    p->v    = calloc(B * KV_DIM,      sizeof(float));
    p->att  = calloc(B * Q_DIM,       sizeof(float));
    p->gate = calloc(B * FFN_SIZE,    sizeof(float));
    p->up   = calloc(B * FFN_SIZE,    sizeof(float));
    p->ffn  = calloc(B * FFN_SIZE,    sizeof(float));
    if (!(p->x && p->proj && p->q && p->k && p->v && p->att && p->gate && p->up && p->ffn)) {
        prefill_work_free(p);
        return 0;
    }
    return 1;
}

/* Process n prompt tokens (n <= PREFILL_BATCH) at positions pos0..pos0+n-1.
   CPU path only. Logits are produced only when want_logits is set. */
static int forward_prefill_batch(GGUFFile *g, const uint32_t *tokens, size_t n,
                                 size_t pos0, float *kc, float *vc, Work *w,
                                 PrefillWork *p, float *logits, uint64_t vocab,
                                 int want_logits)
{
    TensorInfo emb;
    if (!n || n > PREFILL_BATCH || !find_tensor(g, "token_embd.weight", &emb)) return 0;
    for (size_t t = 0; t < n; ++t) {
        if (tokens[t] >= emb.dims[1] ||
            !load_embedding_row(g, &emb, tokens[t], p->x + t * HIDDEN_SIZE)) return 0;
    }
    for (int layer = 0; layer < LAYERS; ++layer) {
        char name[96], q_name[96], k_name[96], v_name[96];

        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", layer);
        if (!vector_named(g, name, w->norm, HIDDEN_SIZE)) return 0;
        for (size_t t = 0; t < n; ++t)
            rms_norm(p->x + t * HIDDEN_SIZE, w->norm, p->proj + t * HIDDEN_SIZE, HIDDEN_SIZE, RMS_EPS);

        snprintf(q_name, sizeof(q_name), "blk.%d.attn_q.weight", layer);
        snprintf(k_name, sizeof(k_name), "blk.%d.attn_k.weight", layer);
        snprintf(v_name, sizeof(v_name), "blk.%d.attn_v.weight", layer);
        if (!batch_mv(g, q_name, p->proj, p->q, n, HIDDEN_SIZE, Q_DIM)) return 0;
        if (!batch_mv(g, k_name, p->proj, p->k, n, HIDDEN_SIZE, KV_DIM)) return 0;
        if (!batch_mv(g, v_name, p->proj, p->v, n, HIDDEN_SIZE, KV_DIM)) return 0;

        snprintf(name, sizeof(name), "blk.%d.attn_q.bias", layer);
        if (!vector_named(g, name, w->bias, Q_DIM)) return 0;
        for (size_t t = 0; t < n; ++t) add_bias(p->q + t * Q_DIM, w->bias, Q_DIM);
        snprintf(name, sizeof(name), "blk.%d.attn_k.bias", layer);
        if (!vector_named(g, name, w->bias, KV_DIM)) return 0;
        for (size_t t = 0; t < n; ++t) add_bias(p->k + t * KV_DIM, w->bias, KV_DIM);
        snprintf(name, sizeof(name), "blk.%d.attn_v.bias", layer);
        if (!vector_named(g, name, w->bias, KV_DIM)) return 0;
        for (size_t t = 0; t < n; ++t) add_bias(p->v + t * KV_DIM, w->bias, KV_DIM);

        for (size_t t = 0; t < n; ++t) {
            rope(p->q + t * Q_DIM, p->k + t * KV_DIM, Q_DIM, KV_DIM, (int)(pos0 + t));
            if (!gqa_single_token(p->q + t * Q_DIM, p->k + t * KV_DIM, p->v + t * KV_DIM,
                                  p->att + t * Q_DIM, kc, vc, (size_t)layer, pos0 + t,
                                  MAX_CONTEXT, w->scores, w->weights)) return 0;
        }

        snprintf(name, sizeof(name), "blk.%d.attn_output.weight", layer);
        if (!batch_mv(g, name, p->att, p->proj, n, Q_DIM, HIDDEN_SIZE)) return 0;
        for (size_t i = 0; i < n * (size_t)HIDDEN_SIZE; ++i) p->x[i] += p->proj[i];

        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", layer);
        if (!vector_named(g, name, w->norm, HIDDEN_SIZE)) return 0;
        for (size_t t = 0; t < n; ++t)
            rms_norm(p->x + t * HIDDEN_SIZE, w->norm, p->proj + t * HIDDEN_SIZE, HIDDEN_SIZE, RMS_EPS);

        snprintf(q_name, sizeof(q_name), "blk.%d.ffn_gate.weight", layer);
        snprintf(k_name, sizeof(k_name), "blk.%d.ffn_up.weight", layer);
        if (!batch_mv(g, q_name, p->proj, p->gate, n, HIDDEN_SIZE, FFN_SIZE)) return 0;
        if (!batch_mv(g, k_name, p->proj, p->up,   n, HIDDEN_SIZE, FFN_SIZE)) return 0;
        for (size_t i = 0; i < n * (size_t)FFN_SIZE; ++i) {
            float z = p->gate[i];
            p->ffn[i] = (z / (1.0f + expf(-z))) * p->up[i];
        }

        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", layer);
        if (!batch_mv(g, name, p->ffn, p->proj, n, FFN_SIZE, HIDDEN_SIZE)) return 0;
        for (size_t i = 0; i < n * (size_t)HIDDEN_SIZE; ++i) p->x[i] += p->proj[i];
    }

    if (want_logits) {
        TensorInfo out;
        if (!vector_named(g, "output_norm.weight", w->norm, HIDDEN_SIZE)) return 0;
        rms_norm(p->x + (n - 1) * HIDDEN_SIZE, w->norm, w->proj, HIDDEN_SIZE, RMS_EPS);
        if (!find_tensor(g, "output.weight", &out) || out.dims[1] != vocab ||
            !tensor_matvec(g, &out, w->proj, logits)) return 0;
    }
    return 1;
}
'@

$loop = @'
    double t_prefill = now_ms();
    const char *no_batch = getenv("BRAIN_NO_BATCH");
    int use_batch = ok && !g_gpu_enabled && !(no_batch && !strcmp(no_batch,"1"));
    PrefillWork pw = {0};
    if (use_batch && !prefill_work_alloc(&pw)) use_batch = 0;
    if (use_batch) {
        size_t done = 0;
        while (ok && done < prompt_count) {
            size_t n = prompt_count - done; if (n > PREFILL_BATCH) n = PREFILL_BATCH;
            if (!forward_prefill_batch(&g,ids+done,n,done,kc,vc,&w,&pw,logits,vocab,done+n==prompt_count)) { ok=0; break; }
            done += n;
            fprintf(stderr,"Prompt: %zu/%zu tokens\n",done,prompt_count);
        }
        prefill_work_free(&pw);
    } else {
        for (size_t i=0; ok && i<prompt_count; ++i) {
            if(!forward_token(&g,ids[i],i,kc,vc,&w,logits,vocab,i+1==prompt_count)) { ok=0; break; }
            fprintf(stderr,"Prompt: %zu/%zu tokens\n",i+1,prompt_count);
        }
    }
'@

$srcPath = "src\brain_runtime_pool.c"
$dstPath = "src\brain_runtime_batch.c"
$utf8 = New-Object System.Text.UTF8Encoding($false)

$lines = [System.IO.File]::ReadAllLines((Resolve-Path $srcPath).Path, $utf8)
$blockLines = $block -split "\r?\n"
$loopLines  = $loop  -split "\r?\n"

$ftIdx = @(0..($lines.Count - 1) | Where-Object {
    $lines[$_] -match '^static int forward_token\(GGUFFile \*g, uint32_t token, size_t pos,' })
$pfIdx = @(0..($lines.Count - 1) | Where-Object {
    $lines[$_] -match 'double t_prefill = now_ms\(\); for \(size_t i=0; ok && i<prompt_count; \+\+i\) \{' })

if ($ftIdx.Count -ne 1) { throw "Expected exactly one forward_token definition, found $($ftIdx.Count). Nothing was written." }
if ($pfIdx.Count -ne 1) { throw "Expected exactly one prefill loop, found $($pfIdx.Count). Nothing was written." }

$ft = $ftIdx[0]
$pf = $pfIdx[0]
if ($ft -ge $pf) { throw "forward_token is not before the prefill loop. Nothing was written." }

if ($lines[$pf + 1] -notmatch 'forward_token\(&g,ids\[i\]' -or
    $lines[$pf + 2] -notmatch 'Prompt: ' -or
    $lines[$pf + 3].Trim() -ne '}') {
    throw "The prefill loop does not look like the expected 4 lines. Nothing was written."
}

$out = New-Object System.Collections.Generic.List[string]
$out.AddRange([string[]]$lines[0..($ft - 1)])
$out.AddRange([string[]]$blockLines)
$out.Add("")
$out.AddRange([string[]]$lines[$ft..($pf - 1)])
$out.AddRange([string[]]$loopLines)
$out.AddRange([string[]]$lines[($pf + 4)..($lines.Count - 1)])

[System.IO.File]::WriteAllLines((Join-Path (Get-Location).Path $dstPath), $out, $utf8)
Write-Host "Wrote $dstPath"
Write-Host ("Inserted batch block before line {0}; replaced the 4-line prefill loop at line {1}." -f ($ft + 1), ($pf + 1))