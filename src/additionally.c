#include "additionally.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// global GPU index: cuda.c
int gpu_index = 0;

// im2col.c
float im2col_get_pixel(float *im, int height, int width, int channels,
    int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

// im2col.c
//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu(float* data_im,
    int channels, int height, int width,
    int ksize, int stride, int pad, float* data_col)
{
    int c, h, w;
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                    im_row, im_col, c_im, pad);
            }
        }
    }
}

// fuse convolutional and batch_norm weights into one convolutional-layer
void yolov2_fuse_conv_batchnorm(network net)
{
    int j;
    for (j = 0; j < net.n; ++j) {
        layer *l = &net.layers[j];

        if (l->type == CONVOLUTIONAL) {
            printf(" Fuse Convolutional layer \t\t l->size = %d  \n", l->size);

            if (l->batch_normalize) {
                int f;
                for (f = 0; f < l->n; ++f)
                {
                    l->biases[f] = l->biases[f] - l->scales[f] * l->rolling_mean[f] / (sqrtf(l->rolling_variance[f]) + .000001f);

                    const size_t filter_size = l->size*l->size*l->c;
                    int i;
                    for (i = 0; i < filter_size; ++i) {
                        int w_index = f*filter_size + i;

                        l->weights[w_index] = l->weights[w_index] * l->scales[f] / (sqrtf(l->rolling_variance[f]) + .000001f);
                    }
                }

                l->batch_normalize = 0;
            }
        }
        else {
            printf(" Skip layer: %d \n", l->type);
        }
    }
}

// -------------- XNOR-net ------------

void binarize_weights(float *weights, int n, int size, float *binary)
{
    int i, f;
    for (f = 0; f < n; ++f) {
        float mean = 0;
        for (i = 0; i < size; ++i) {
            mean += fabs(weights[f*size + i]);
        }
        mean = mean / size;
        for (i = 0; i < size; ++i) {
            binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        }
    }
}

void binarize_cpu(float *input, int n, float *binary)
{
    int i;
    for (i = 0; i < n; ++i) {
        binary[i] = (input[i] > 0) ? 1 : -1;
    }
}

static inline unsigned char xnor(unsigned char a, unsigned char b) {
    //return a == b;
    return !(a^b);
}

void get_mean_array(float *src, size_t size, size_t filters, float *mean_arr) {
    size_t i, counter;
    counter = 0;
    for (i = 0; i < size; i += size / filters) {
        mean_arr[counter++] = fabs(src[i]);
    }
}

void binary_align_weights(convolutional_layer *l)
{
    int m = l->n;
    int k = l->size*l->size*l->c;
    size_t new_lda = k + (l->lda_align - k % l->lda_align); // (k / 8 + 1) * 8;
    l->new_lda = new_lda;

    binarize_weights(l->weights, m, k, l->binary_weights);

    size_t align_weights_size = new_lda * m;
    l->align_bit_weights_size = align_weights_size / 8 + 1;
    float *align_weights = calloc(align_weights_size, sizeof(float));
    l->align_bit_weights = calloc(l->align_bit_weights_size, sizeof(char));

    size_t i, j;
    // align A without transpose
    for (i = 0; i < m; ++i) {
        for (j = 0; j < k; ++j) {
            align_weights[i*new_lda + j] = l->binary_weights[i*k + j];
        }
    }
    float_to_bit(align_weights, l->align_bit_weights, align_weights_size);

    l->mean_arr = calloc(l->n, sizeof(float));
    get_mean_array(align_weights, align_weights_size, l->n, l->mean_arr);

    free(align_weights);
}

void calculate_binary_weights(network net)
{
    int j;
    for (j = 0; j < net.n; ++j) {
        layer *l = &net.layers[j];

        if (l->type == CONVOLUTIONAL) {
            //printf(" Merges Convolutional-%d and batch_norm \n", j);

            if (l->xnor) {
                //printf("\n %d \n", j);
                l->lda_align = 256; // 256bit for AVX2

                binary_align_weights(l);

                if (net.layers[j].use_bin_output) {
                    l->activation = LINEAR;
                }
            }

//			if(m_dbg) 
//			{
//				layer l1 = net.layers[j];
//				if (l1.dontload) continue;
//				save_convolutional_weights(l1, j);
//			}
        }
    }
}

static inline void set_bit(unsigned char *const dst, size_t index) {
    size_t dst_i = index / 8;
    int dst_shift = index % 8;
    dst[dst_i] |= 1 << dst_shift;
}

static inline unsigned char get_bit(unsigned char const*const src, size_t index) {
    size_t src_i = index / 8;
    int src_shift = index % 8;
    unsigned char val = (src[src_i] & (1 << src_shift)) > 0;
    return val;
}

uint8_t reverse_8_bit(uint8_t a) {
    return ((a * 0x0802LU & 0x22110LU) | (a * 0x8020LU & 0x88440LU)) * 0x10101LU >> 16;
}

uint32_t reverse_32_bit(uint32_t a)
{
    return (reverse_8_bit(a >> 24) << 0) |
        (reverse_8_bit(a >> 16) << 8) |
        (reverse_8_bit(a >> 8) << 16) |
        (reverse_8_bit(a >> 0) << 24);
}

#define swap(a0, a1, j, m) t = (a0 ^ (a1 >>j)) & m; a0 = a0 ^ t; a1 = a1 ^ (t << j);

void transpose32_optimized(uint32_t A[32]) {
    int j, k;
    unsigned m, t;

    j = 16;
    m = 0x0000FFFF;
    for (k = 0; k < 32; k = (k + j + 1) & ~j) { swap(A[k], A[k + j], j, m); }

    j = 8;
    m = 0x00ff00ff;
    for (k = 0; k < 32; k = (k + j + 1) & ~j) { swap(A[k], A[k + j], j, m); }

    j = 4;
    m = 0x0f0f0f0f;
    for (k = 0; k < 32; k = (k + j + 1) & ~j) { swap(A[k], A[k + j], j, m); }

    j = 2;
    m = 0x33333333;
    for (k = 0; k < 32; k = (k + j + 1) & ~j) { swap(A[k], A[k + j], j, m); }

    j = 1;
    m = 0x55555555;
    for (k = 0; k < 32; k = (k + j + 1) & ~j) { swap(A[k], A[k + j], j, m); }

    // reverse Y
    for (j = 0; j < 16; ++j) {
        uint32_t tmp = A[j];
        A[j] = reverse_32_bit(A[31 - j]);
        A[31 - j] = reverse_32_bit(tmp);
    }
}

void transpose_32x32_bits_reversed_diagonale(uint32_t *A, uint32_t *B, int m, int n)
{
    unsigned A_tmp[32];
    int i;
    #pragma unroll
    for (i = 0; i < 32; ++i) A_tmp[i] = A[i * m];
    transpose32_optimized(A_tmp);
    #pragma unroll
    for (i = 0; i < 32; ++i) B[i*n] = A_tmp[i];
}

// transpose by 32-bit
void transpose_bin(uint32_t *A, uint32_t *B, const int n, const int m,
    const int lda, const int ldb, const int block_size)
{
    int i;
    #pragma omp parallel for
    for (i = 0; i < n; i += 32) {
        int j;
        for (j = 0; j < m; j += 32) {
            int a_index = i*lda + j;
            int b_index = j*ldb + i;
            transpose_32x32_bits_reversed_diagonale(&A[a_index / 32], &B[b_index / 32], lda / 32, ldb / 32);
        }
        for (; j < m; ++j) {
            if (get_bit(A, i*lda + j)) set_bit(B, j*ldb + i);
        }
    }
}

void activate_array_cpu_custom(float *x, const int n, const ACTIVATION a)
{
    int i = 0;
    if (a == LINEAR)  {}
    else {
        for (i = 0; i < n; ++i) {
            x[i] = activate(x[i], a);
        }
    }
}


void forward_maxpool_layer_avx(float *src, float *dst, int *indexes, int size, int w, int h, int out_w, int out_h, int c,
    int pad, int stride, int batch)
{
    int b, k;
    const int w_offset = -pad / 2;
    const int h_offset = -pad / 2;

    for (b = 0; b < batch; ++b) {
        #pragma omp parallel for
        for (k = 0; k < c; ++k) {
            int i, j, m, n;
            for (i = 0; i < out_h; ++i) {
                for (j = 0; j < out_w; ++j) {
                    int out_index = j + out_w*(i + out_h*(k + c*b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    for (n = 0; n < size; ++n) {
                        for (m = 0; m < size; ++m) {
                            int cur_h = h_offset + i*stride + n;
                            int cur_w = w_offset + j*stride + m;
                            int index = cur_w + w*(cur_h + h*(k + b*c));
                            int valid = (cur_h >= 0 && cur_h < h &&
                                cur_w >= 0 && cur_w < w);
                            float val = (valid != 0) ? src[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;
                            max = (val > max) ? val : max;
                        }
                    }
                    dst[out_index] = max;
                    indexes[out_index] = max_i;
                }
            }
        }
    }
}

void float_to_bit(float *src, unsigned char *dst, size_t size)
{
    size_t dst_size = size / 8 + 1;
    memset(dst, 0, dst_size);

    size_t i;
    char *byte_arr = calloc(size, sizeof(char));
    for (i = 0; i < size; ++i) {
        if (src[i] > 0) byte_arr[i] = 1;
    }

    //for (i = 0; i < size; ++i) {
    //    dst[i / 8] |= byte_arr[i] << (i % 8);
    //}

    for (i = 0; i < size; i += 8) {
        char dst_tmp = 0;
        dst_tmp |= byte_arr[i + 0] << 0;
        dst_tmp |= byte_arr[i + 1] << 1;
        dst_tmp |= byte_arr[i + 2] << 2;
        dst_tmp |= byte_arr[i + 3] << 3;
        dst_tmp |= byte_arr[i + 4] << 4;
        dst_tmp |= byte_arr[i + 5] << 5;
        dst_tmp |= byte_arr[i + 6] << 6;
        dst_tmp |= byte_arr[i + 7] << 7;
        dst[i / 8] = dst_tmp;
    }
    free(byte_arr);
}

// -------------- utils.c --------------


// utils.c
void error(const char *s)
{
    perror(s);
    assert(0);
    exit(-1);
}

// utils.c
void malloc_error()
{
    fprintf(stderr, "Malloc error\n");
    exit(-1);
}

// utils.c
void file_error(char *s)
{
    fprintf(stderr, "Couldn't open file: %s\n", s);
    exit(0);
}

// utils.c
char *fgetl(FILE *fp)
{
    if (feof(fp)) return 0;
    size_t size = 512;
    char *line = malloc(size * sizeof(char));
    if (!fgets(line, size, fp)) {
        free(line);
        return 0;
    }

    size_t curr = strlen(line);

    while ((line[curr - 1] != '\n') && !feof(fp)) {
        if (curr == size - 1) {
            size *= 2;
            line = realloc(line, size * sizeof(char));
            if (!line) {
                printf("%ld\n", (int long)size);
                malloc_error();
            }
        }
        size_t readsize = size - curr;
        if (readsize > INT_MAX) readsize = INT_MAX - 1;
        fgets(&line[curr], readsize, fp);
        curr = strlen(line);
    }
    if (line[curr - 1] == '\n') line[curr - 1] = '\0';

    return line;
}

// utils.c
void del_arg(int argc, char **argv, int index)
{
    int i;
    for (i = index; i < argc - 1; ++i) argv[i] = argv[i + 1];
    argv[i] = 0;
}

// utils.c
int find_arg(int argc, char* argv[], char *arg)
{
    int i;
    for (i = 0; i < argc; ++i) {
        if (!argv[i]) continue;
        if (0 == strcmp(argv[i], arg)) {
            del_arg(argc, argv, i);
            return 1;
        }
    }
    return 0;
}

// utils.c
int find_int_arg(int argc, char **argv, char *arg, int def)
{
    int i;
    for (i = 0; i < argc - 1; ++i) {
        if (!argv[i]) continue;
        if (0 == strcmp(argv[i], arg)) {
            def = atoi(argv[i + 1]);
            del_arg(argc, argv, i);
            del_arg(argc, argv, i);
            break;
        }
    }
    return def;
}

// utils.c
float find_float_arg(int argc, char **argv, char *arg, float def)
{
    int i;
    for (i = 0; i < argc - 1; ++i) {
        if (!argv[i]) continue;
        if (0 == strcmp(argv[i], arg)) {
            def = atof(argv[i + 1]);
            del_arg(argc, argv, i);
            del_arg(argc, argv, i);
            break;
        }
    }
    return def;
}

// utils.c
char *find_char_arg(int argc, char **argv, char *arg, char *def)
{
    int i;
    for (i = 0; i < argc - 1; ++i) {
        if (!argv[i]) continue;
        if (0 == strcmp(argv[i], arg)) {
            def = argv[i + 1];
            del_arg(argc, argv, i);
            del_arg(argc, argv, i);
            break;
        }
    }
    return def;
}


// utils.c
void strip(char *s)
{
    size_t i;
    size_t len = strlen(s);
    size_t offset = 0;
    for (i = 0; i < len; ++i) {
        char c = s[i];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++offset;
        else s[i - offset] = c;
    }
    s[len - offset] = '\0';
}

// utils.c
void list_insert(list *l, void *val)
{
    node *new = malloc(sizeof(node));
    new->val = val;
    new->next = 0;

    if (!l->back) {
        l->front = new;
        new->prev = 0;
    }
    else {
        l->back->next = new;
        new->prev = l->back;
    }
    l->back = new;
    ++l->size;
}


// utils.c
float rand_uniform(float min, float max)
{
    if (max < min) {
        float swap = min;
        min = max;
        max = swap;
    }
    return ((float)rand() / RAND_MAX * (max - min)) + min;
}

// utils.c
float rand_scale(float s)
{
    float scale = rand_uniform(1, s);
    if (rand() % 2) return scale;
    return 1. / scale;
}

// utils.c
int rand_int(int min, int max)
{
    if (max < min) {
        int s = min;
        min = max;
        max = s;
    }
    int r = (rand() % (max - min + 1)) + min;
    return r;
}

// utils.c
int constrain_int(int a, int min, int max)
{
    if (a < min) return min;
    if (a > max) return max;
    return a;
}

// utils.c
float dist_array(float *a, float *b, int n, int sub)
{
    int i;
    float sum = 0;
    for (i = 0; i < n; i += sub) sum += powf(a[i] - b[i], 2);
    return sqrt(sum);
}

// utils.c
float mag_array(float *a, int n)
{
    int i;
    float sum = 0;
    for (i = 0; i < n; ++i) {
        sum += a[i] * a[i];
    }
    return sqrt(sum);
}

// utils.c
int max_index(float *a, int n)
{
    if (n <= 0) return -1;
    int i, max_i = 0;
    float max = a[0];
    for (i = 1; i < n; ++i) {
        if (a[i] > max) {
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}

// utils.c
void free_ptrs(void **ptrs, int n)
{
    int i;
    for (i = 0; i < n; ++i) free(ptrs[i]);
    free(ptrs);
}

// -------------- list.c --------------


// list.c
list *make_list()
{
    list *l = malloc(sizeof(list));
    l->size = 0;
    l->front = 0;
    l->back = 0;
    return l;
}


// list.c
list *get_paths(char *filename)
{
    char *path;
    FILE *file = fopen(filename, "r");
    if (!file) file_error(filename);
    list *lines = make_list();
    while ((path = fgetl(file))) {
        list_insert(lines, path);
    }
    fclose(file);
    return lines;
}


// list.c
void **list_to_array(list *l)
{
    void **a = calloc(l->size, sizeof(void*));
    int count = 0;
    node *n = l->front;
    while (n) {
        a[count++] = n->val;
        n = n->next;
    }
    return a;
}

// list.c
void free_node(node *n)
{
    node *next;
    while (n) {
        next = n->next;
        free(n);
        n = next;
    }
}

// list.c
void free_list(list *l)
{
    free_node(l->front);
    free(l);
}

// -------------- network.c --------------

// network.c
float *get_network_output(network net)
{
    int i;
    for (i = net.n - 1; i > 0; --i) if (net.layers[i].type != COST) break;
    return net.layers[i].output;
}

// network.c
int get_network_output_size(network net)
{
    int i;
    for (i = net.n - 1; i > 0; --i) if (net.layers[i].type != COST) break;
    return net.layers[i].outputs;
}

// network.c
network make_network(int n)
{
    network net = { 0 };
    net.n = n;
    net.layers = calloc(net.n, sizeof(layer));
    net.seen = calloc(1, sizeof(uint64_t));
    return net;
}

void free_network(network net)
{
    int i;
    for (i = 0; i < net.n; ++i) {
        free_layer(net.layers[i]);
    }
    free(net.layers);

    free(net.scales);
    free(net.steps);
    free(net.seen);

    free(net.workspace);
}

// network.c
void set_batch_network(network *net, int b)
{
    net->batch = b;
    int i;
    for (i = 0; i < net->n; ++i) {
        layer l = net->layers[i];
        l.batch = b;
    }
}

// -------------- layer.c --------------
void free_layer(layer l)
{
    if (l.type == DROPOUT) {
        if (l.rand)           free(l.rand);
        return;
    }
    if (l.cweights)           free(l.cweights);
    if (l.indexes)            free(l.indexes);
    if (l.input_layers)       free(l.input_layers);
    if (l.input_sizes)        free(l.input_sizes);
    if (l.map)                free(l.map);
    if (l.rand)               free(l.rand);
    if (l.cost)               free(l.cost);
    if (l.state)              free(l.state);
    if (l.prev_state)         free(l.prev_state);
    if (l.forgot_state)       free(l.forgot_state);
    if (l.forgot_delta)       free(l.forgot_delta);
    if (l.state_delta)        free(l.state_delta);
    if (l.concat)             free(l.concat);
    if (l.concat_delta)       free(l.concat_delta);
    if (l.binary_weights)     free(l.binary_weights);
    if (l.biases)             free(l.biases);
    if (l.biases_quant)       free(l.biases_quant);
    //if (l.bias_updates)       free(l.bias_updates);
    if (l.scales)             free(l.scales);
    //if (l.scale_updates)      free(l.scale_updates);
    if (l.weights)            free(l.weights);
    if (l.weights_int8)       free(l.weights_int8);
    if (l.align_bit_weights)  free(l.align_bit_weights);
    if (l.mean_arr)           free(l.mean_arr);
    //if (l.weight_updates)     free(l.weight_updates);
    //if (l.delta)              free(l.delta);
    if (l.output)             free(l.output);
    if (l.squared)            free(l.squared);
    if (l.norms)              free(l.norms);
    if (l.spatial_mean)       free(l.spatial_mean);
    if (l.mean)               free(l.mean);
    if (l.variance)           free(l.variance);
    //if (l.mean_delta)         free(l.mean_delta);
    //if (l.variance_delta)     free(l.variance_delta);
    if (l.rolling_mean)       free(l.rolling_mean);
    if (l.rolling_variance)   free(l.rolling_variance);
    if (l.x)                  free(l.x);
    if (l.x_norm)             free(l.x_norm);
    if (l.m)                  free(l.m);
    if (l.v)                  free(l.v);
    if (l.z_cpu)              free(l.z_cpu);
    if (l.r_cpu)              free(l.r_cpu);
    if (l.h_cpu)              free(l.h_cpu);
    if (l.binary_input)       free(l.binary_input);
}


// -------------- softmax_layer.c --------------

// softmax_layer.c
softmax_layer make_softmax_layer(int batch, int inputs, int groups)
{
    assert(inputs%groups == 0);
    fprintf(stderr, "softmax                                        %4d\n", inputs);
    softmax_layer l = { 0 };
    l.type = SOFTMAX;
    l.batch = batch;
    l.groups = groups;
    l.inputs = inputs;
    l.outputs = inputs;
    l.output = calloc(inputs*batch, sizeof(float));
    return l;
}

// -------------- region_layer.c --------------

//  region_layer.c
region_layer make_region_layer(int batch, int w, int h, int n, int classes, int coords)
{
    region_layer l = { 0 };
    l.type = REGION;

    l.n = n;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.classes = classes;
    l.coords = coords;
    l.cost = calloc(1, sizeof(float));
    l.biases = calloc(n * 2, sizeof(float));
    //l.bias_updates = calloc(n * 2, sizeof(float));
    l.outputs = h*w*n*(classes + coords + 1);
    l.inputs = l.outputs;
    l.truths = 30 * (5);
    //l.delta = calloc(batch*l.outputs, sizeof(float));
    l.output = calloc(batch*l.outputs, sizeof(float));
    int i;
    for (i = 0; i < n * 2; ++i) {
        l.biases[i] = .5;
    }

    // commented only for this custom version of Yolo v2
    //l.forward = forward_region_layer;
    //l.backward = backward_region_layer;
    fprintf(stderr, "detection\n");
    srand(0);

    return l;
}


// -------------- maxpool_layer.c --------------

// maxpool_layer.c
maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding)
{
    maxpool_layer l = { 0 };
    l.type = MAXPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.pad = padding;
    l.out_w = (w + padding - size) / stride + 1;
    l.out_h = (h + padding - size) / stride + 1;
    l.out_c = c;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    l.size = size;
    l.stride = stride;
    int output_size = l.out_h * l.out_w * l.out_c * batch;
    l.indexes = calloc(output_size, sizeof(int));
    l.output = calloc(output_size, sizeof(float));
    l.output_int8 = calloc(output_size, sizeof(int8_t));
    //l.delta = calloc(output_size, sizeof(float));
    // commented only for this custom version of Yolo v2
    //l.forward = forward_maxpool_layer;
    //l.backward = backward_maxpool_layer;
    fprintf(stderr, "max          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}


// -------------- convolutional_layer.c --------------

// convolutional_layer.c
size_t get_workspace_size(layer l) {
    if (l.xnor) return (size_t)l.bit_align*l.size*l.size*l.c * sizeof(float);
    return (size_t)l.out_h*l.out_w*l.size*l.size*l.c * sizeof(float);
}

int convolutional_out_height(convolutional_layer l)
{
    return (l.h + 2 * l.pad - l.size) / l.stride + 1;
}

int convolutional_out_width(convolutional_layer l)
{
    return (l.w + 2 * l.pad - l.size) / l.stride + 1;
}


// convolutional_layer.c
convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam, int quantized, int use_bin_output)
{
    int i;
    convolutional_layer l = { 0 };
    l.type = CONVOLUTIONAL;
    l.quantized = quantized;

    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.binary = binary;
    l.xnor = xnor;
    l.use_bin_output = use_bin_output;
    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = padding;
    l.batch_normalize = batch_normalize;

    l.weights = calloc(c*n*size*size, sizeof(float));
    l.weights_int8 = calloc(c*n*size*size, sizeof(int8_t));
    //l.weight_updates = calloc(c*n*size*size, sizeof(float));

    l.biases = calloc(n, sizeof(float));
    l.biases_quant = calloc(n, sizeof(float));
    //l.bias_updates = calloc(n, sizeof(float));

    // float scale = 1./sqrt(size*size*c);
    float scale = sqrt(2. / (size*size*c));
    for (i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_uniform(-1, 1);
    int out_h = convolutional_out_height(l);
    int out_w = convolutional_out_width(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.output = calloc(l.batch*l.outputs, sizeof(float));
    l.output_int8 = calloc(l.batch*l.outputs, sizeof(int8_t));

	if (binary) {
        l.binary_weights = calloc(c*n*size*size, sizeof(float));
        l.cweights = calloc(c*n*size*size, sizeof(char));
        l.scales = calloc(n, sizeof(float));
    }
    if (xnor) {
        l.binary_weights = calloc(c*n*size*size, sizeof(float));
        l.binary_input = calloc(l.inputs*l.batch, sizeof(float));

        int align = 32;// 8;
        int src_align = l.out_h*l.out_w;
        l.bit_align = src_align + (align - src_align % align);
    }

    if (batch_normalize) {
        l.scales = calloc(n, sizeof(float));
        //l.scale_updates = calloc(n, sizeof(float));
        for (i = 0; i < n; ++i) {
            l.scales[i] = 1;
        }

        l.mean = calloc(n, sizeof(float));
        l.variance = calloc(n, sizeof(float));

        //l.mean_delta = calloc(n, sizeof(float));
        //l.variance_delta = calloc(n, sizeof(float));

        l.rolling_mean = calloc(n, sizeof(float));
        l.rolling_variance = calloc(n, sizeof(float));
        l.x = calloc(l.batch*l.outputs, sizeof(float));
        l.x_norm = calloc(l.batch*l.outputs, sizeof(float));
    }
    if (adam) {
        l.adam = 1;
        l.m = calloc(c*n*size*size, sizeof(float));
        l.v = calloc(c*n*size*size, sizeof(float));
    }

    l.workspace_size = get_workspace_size(l);
    l.activation = activation;

    l.bflops = (2.0 * l.n * l.size*l.size*l.c * l.out_h*l.out_w) / 1000000000.;
    if (l.xnor && l.use_bin_output) fprintf(stderr, "convXB");
    else if (l.xnor) fprintf(stderr, "convX ");
    else fprintf(stderr, "conv  ");
    fprintf(stderr, "%5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d %5.3f BF\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c, l.bflops);

    return l;
}

// -------------- image.c --------------

// image.c
void rgbgr_image(image im)
{
    int i;
    for (i = 0; i < im.w*im.h; ++i) {
        float swap = im.data[i];
        im.data[i] = im.data[i + im.w*im.h * 2];
        im.data[i + im.w*im.h * 2] = swap;
    }
}

// image.c
image make_empty_image(int w, int h, int c)
{
    image out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}

// image.c
void free_image(image m)
{
    if (m.data) {
        free(m.data);
    }
}

// image.c
void draw_box(image a, int x1, int y1, int x2, int y2, float r, float g, float b)
{
    //normalize_image(a);
    int i;
    if (x1 < 0) x1 = 0;
    if (x1 >= a.w) x1 = a.w - 1;
    if (x2 < 0) x2 = 0;
    if (x2 >= a.w) x2 = a.w - 1;

    if (y1 < 0) y1 = 0;
    if (y1 >= a.h) y1 = a.h - 1;
    if (y2 < 0) y2 = 0;
    if (y2 >= a.h) y2 = a.h - 1;

    for (i = x1; i <= x2; ++i) {
        a.data[i + y1*a.w + 0 * a.w*a.h] = r;
        a.data[i + y2*a.w + 0 * a.w*a.h] = r;

        a.data[i + y1*a.w + 1 * a.w*a.h] = g;
        a.data[i + y2*a.w + 1 * a.w*a.h] = g;

        a.data[i + y1*a.w + 2 * a.w*a.h] = b;
        a.data[i + y2*a.w + 2 * a.w*a.h] = b;
    }
    for (i = y1; i <= y2; ++i) {
        a.data[x1 + i*a.w + 0 * a.w*a.h] = r;
        a.data[x2 + i*a.w + 0 * a.w*a.h] = r;

        a.data[x1 + i*a.w + 1 * a.w*a.h] = g;
        a.data[x2 + i*a.w + 1 * a.w*a.h] = g;

        a.data[x1 + i*a.w + 2 * a.w*a.h] = b;
        a.data[x2 + i*a.w + 2 * a.w*a.h] = b;
    }
}

// image.c
void draw_box_width(image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b)
{
    int i;
    for (i = 0; i < w; ++i) {
        draw_box(a, x1 + i, y1 + i, x2 - i, y2 - i, r, g, b);
    }
}

// image.c
image make_image(int w, int h, int c)
{
    image out = make_empty_image(w, h, c);
    out.data = calloc(h*w*c, sizeof(float));
    return out;
}

// image.c
float get_pixel(image m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y*m.w + x];
}

// image.c
void set_pixel(image m, int x, int y, int c, float val)
{
    if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c) return;
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] = val;
}

// image.c
void add_pixel(image m, int x, int y, int c, float val)
{
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] += val;
}

// image.c
image resize_image(image im, int w, int h)
{
    image resized = make_image(w, h, im.c);
    image part = make_image(w, im.h, im.c);
    int r, c, k;
    float w_scale = (float)(im.w - 1) / (w - 1);
    float h_scale = (float)(im.h - 1) / (h - 1);
    for (k = 0; k < im.c; ++k) {
        for (r = 0; r < im.h; ++r) {
            for (c = 0; c < w; ++c) {
                float val = 0;
                if (c == w - 1 || im.w == 1) {
                    val = get_pixel(im, im.w - 1, r, k);
                }
                else {
                    float sx = c*w_scale;
                    int ix = (int)sx;
                    float dx = sx - ix;
                    val = (1 - dx) * get_pixel(im, ix, r, k) + dx * get_pixel(im, ix + 1, r, k);
                }
                set_pixel(part, c, r, k, val);
            }
        }
    }
    for (k = 0; k < im.c; ++k) {
        for (r = 0; r < h; ++r) {
            float sy = r*h_scale;
            int iy = (int)sy;
            float dy = sy - iy;
            for (c = 0; c < w; ++c) {
                float val = (1 - dy) * get_pixel(part, c, iy, k);
                set_pixel(resized, c, r, k, val);
            }
            if (r == h - 1 || im.h == 1) continue;
            for (c = 0; c < w; ++c) {
                float val = dy * get_pixel(part, c, iy + 1, k);
                add_pixel(resized, c, r, k, val);
            }
        }
    }

    free_image(part);
    return resized;
}

// image.c
image load_image(char *filename, int w, int h, int c)
{
#ifdef OPENCV
    image out = load_image_cv(filename, c);
#else
    image out = load_image_stb(filename, c);
#endif

    if ((h && w) && (h != out.h || w != out.w)) {
        image resized = resize_image(out, w, h);
        free_image(out);
        out = resized;
    }
    return out;
}

// image.c
image load_image_stb(char *filename, int channels)
{
    int w, h, c;
    unsigned char *data = stbi_load(filename, &w, &h, &c, channels);
    if (!data) {
        fprintf(stderr, "Cannot load image \"%s\"\nSTB Reason: %s\n", filename, stbi_failure_reason());
        exit(0);
    }
    if (channels) c = channels;
    int i, j, k;
    image im = make_image(w, h, c);
    for (k = 0; k < c; ++k) {
        for (j = 0; j < h; ++j) {
            for (i = 0; i < w; ++i) {
                int dst_index = i + w*j + w*h*k;
                int src_index = k + c*i + c*w*j;
                im.data[dst_index] = (float)data[src_index] / 255.;
            }
        }
    }
    free(data);
    return im;
}


#ifdef OPENCV

// image.c
image ipl_to_image(IplImage* src)
{
    unsigned char *data = (unsigned char *)src->imageData;
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    int step = src->widthStep;
    image out = make_image(w, h, c);
    int i, j, k, count = 0;;

    for (k = 0; k < c; ++k) {
        for (i = 0; i < h; ++i) {
            for (j = 0; j < w; ++j) {
                out.data[count++] = data[i*step + j*c + k] / 255.;
            }
        }
    }
    return out;
}

// image.c
image load_image_cv(char *filename, int channels)
{
    IplImage* src = 0;
    int flag = -1;
    if (channels == 0) flag = -1;
    else if (channels == 1) flag = 0;
    else if (channels == 3) flag = 1;
    else {
        fprintf(stderr, "OpenCV can't force load with %d channels\n", channels);
    }

    if ((src = cvLoadImage(filename, flag)) == 0)
    {
        fprintf(stderr, "Cannot load image \"%s\"\n", filename);
        char buff[256];
        sprintf(buff, "echo %s >> bad.list", filename);
        system(buff);
        return make_image(10, 10, 3);
        //exit(0);
    }
    image out = ipl_to_image(src);
    cvReleaseImage(&src);
    rgbgr_image(out);
    return out;
}
#endif    // OPENCV

// image.c
image copy_image(image p)
{
    image copy = p;
    copy.data = calloc(p.h*p.w*p.c, sizeof(float));
    memcpy(copy.data, p.data, p.h*p.w*p.c * sizeof(float));
    return copy;
}

// image.c
void constrain_image(image im)
{
    int i;
    for (i = 0; i < im.w*im.h*im.c; ++i) {
        if (im.data[i] < 0) im.data[i] = 0;
        if (im.data[i] > 1) im.data[i] = 1;
    }
}

#ifdef OPENCV
// image.c
void show_image_cv(image p, const char *name)
{
    int x, y, k;
    image copy = copy_image(p);
    constrain_image(copy);
    if (p.c == 3) rgbgr_image(copy);
    char buff[256];
    sprintf(buff, "%s", name);

    IplImage *disp = cvCreateImage(cvSize(p.w, p.h), IPL_DEPTH_8U, p.c);
    int step = disp->widthStep;
    cvNamedWindow(buff, CV_WINDOW_NORMAL);
    for (y = 0; y < p.h; ++y) {
        for (x = 0; x < p.w; ++x) {
            for (k = 0; k < p.c; ++k) {
                disp->imageData[y*step + x*p.c + k] = (unsigned char)(get_pixel(copy, x, y, k) * 255);
            }
        }
    }
    free_image(copy);
    cvShowImage(buff, disp);

    cvReleaseImage(&disp);
}

// image.c
void show_image_cv_ipl(IplImage *disp, const char *name)
{
    if (disp == NULL) return;
    char buff[256];
    sprintf(buff, "%s", name);
    cvNamedWindow(buff, CV_WINDOW_NORMAL);
    cvShowImage(buff, disp);
}
#endif

// image.c
void save_image_png(image im, const char *name)
{
    char buff[256];
    sprintf(buff, "%s.png", name);
    unsigned char *data = calloc(im.w*im.h*im.c, sizeof(char));
    int i, k;
    for (k = 0; k < im.c; ++k) {
        for (i = 0; i < im.w*im.h; ++i) {
            data[i*im.c + k] = (unsigned char)(255 * im.data[i + k*im.w*im.h]);
        }
    }
    int success = stbi_write_png(buff, im.w, im.h, im.c, data, im.w*im.c);
    free(data);
    if (!success) fprintf(stderr, "Failed to write image %s\n", buff);
}


// image.c
void show_image(image p, const char *name)
{
#ifdef OPENCV
    show_image_cv(p, name);
#else
    fprintf(stderr, "Not compiled with OpenCV, saving to %s.png instead\n", name);
    save_image_png(p, name);
#endif
}

// image.c
float get_color(int c, int x, int max)
{
    static float colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };
    float ratio = ((float)x / max) * 5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    float r = (1 - ratio) * colors[i][c] + ratio*colors[j][c];
    //printf("%f\n", r);
    return r;
}


// -------------- option_list.c --------------------

// option_list.c
typedef struct {
    char *key;
    char *val;
    int used;
} kvp;

// option_list.c
void option_insert(list *l, char *key, char *val)
{
    kvp *p = malloc(sizeof(kvp));
    p->key = key;
    p->val = val;
    p->used = 0;
    list_insert(l, p);
}

// option_list.c
int read_option(char *s, list *options)
{
    size_t i;
    size_t len = strlen(s);
    char *val = 0;
    for (i = 0; i < len; ++i) {
        if (s[i] == '=') {
            s[i] = '\0';
            val = s + i + 1;
            break;
        }
    }
    if (i == len - 1) return 0;
    char *key = s;
    option_insert(options, key, val);
    return 1;
}

// option_list.c
list *read_data_cfg(char *filename)
{
    FILE *file = fopen(filename, "r");
    if (file == 0) file_error(filename);
    char *line;
    int nu = 0;
    list *options = make_list();
    while ((line = fgetl(file)) != 0) {
        ++nu;
        strip(line);
        switch (line[0]) {
        case '\0':
        case '#':
        case ';':
            free(line);
            break;
        default:
            if (!read_option(line, options)) {
                fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                free(line);
            }
            break;
        }
    }
    fclose(file);
    return options;
}

// option_list.c
void option_unused(list *l)
{
    node *n = l->front;
    while (n) {
        kvp *p = (kvp *)n->val;
        if (!p->used) {
            fprintf(stderr, "Unused field: '%s = %s'\n", p->key, p->val);
        }
        n = n->next;
    }
}

// option_list.c
char *option_find(list *l, char *key)
{
    node *n = l->front;
    while (n) {
        kvp *p = (kvp *)n->val;
        if (strcmp(p->key, key) == 0) {
            p->used = 1;
            return p->val;
        }
        n = n->next;
    }
    return 0;
}

// option_list.c
char *option_find_str(list *l, char *key, char *def)
{
    char *v = option_find(l, key);
    if (v) return v;
    if (def) fprintf(stderr, "%s: Using default '%s'\n", key, def);
    return def;
}

// option_list.c
int option_find_int(list *l, char *key, int def)
{
    char *v = option_find(l, key);
    if (v) return atoi(v);
    fprintf(stderr, "%s: Using default '%d'\n", key, def);
    return def;
}

// option_list.c
int option_find_int_quiet(list *l, char *key, int def)
{
    char *v = option_find(l, key);
    if (v) return atoi(v);
    return def;
}

// option_list.c
float option_find_float_quiet(list *l, char *key, float def)
{
    char *v = option_find(l, key);
    if (v) return atof(v);
    return def;
}

// option_list.c
float option_find_float(list *l, char *key, float def)
{
    char *v = option_find(l, key);
    if (v) return atof(v);
    fprintf(stderr, "%s: Using default '%lf'\n", key, def);
    return def;
}


// -------------- parser.c --------------------

// parser.c
typedef struct size_params {
    int quantized;
    int batch;
    int inputs;
    int h;
    int w;
    int c;
    int index;
    int time_steps;
    network net;
} size_params;

// parser.c
typedef struct {
    char *type;
    list *options;
}section;

// parser.c
list *read_cfg(char *filename)
{
    FILE *file = fopen(filename, "r");
    if (file == 0) file_error(filename);
    char *line;
    int nu = 0;
    list *sections = make_list();
    section *current = 0;
    while ((line = fgetl(file)) != 0) {
        ++nu;
        strip(line);
        switch (line[0]) {
        case '[':
            current = malloc(sizeof(section));
            list_insert(sections, current);
            current->options = make_list();
            current->type = line;
            break;
        case '\0':
        case '#':
        case ';':
            free(line);
            break;
        default:
            if (!read_option(line, current->options)) {
                fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                free(line);
            }
            break;
        }
    }
    fclose(file);
    return sections;
}

// parser.c
void load_convolutional_weights_cpu(layer l, FILE *fp)
{
    int num = l.n*l.c*l.size*l.size;
    fread(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize && (!l.dontloadscales)) {
        fread(l.scales, sizeof(float), l.n, fp);
        fread(l.rolling_mean, sizeof(float), l.n, fp);
        fread(l.rolling_variance, sizeof(float), l.n, fp);
    }
    fread(l.weights, sizeof(float), num, fp);
}

// parser.c
void load_weights_upto_cpu(network *net, char *filename, int cutoff)
{
    fprintf(stderr, "Loading weights from %s...", filename);
    fflush(stdout);
    FILE *fp = fopen(filename, "rb");
    if (!fp) file_error(filename);

    int major;
    int minor;
    int revision;
    fread(&major, sizeof(int), 1, fp);
    fread(&minor, sizeof(int), 1, fp);
    fread(&revision, sizeof(int), 1, fp);
    if ((major * 10 + minor) >= 2) {
        fread(net->seen, sizeof(uint64_t), 1, fp);
    }
    else {
        int iseen = 0;
        fread(&iseen, sizeof(int), 1, fp);
        *net->seen = iseen;
    }
    //int transpose = (major > 1000) || (minor > 1000);

    int i;
    for (i = 0; i < net->n && i < cutoff; ++i) {
        layer l = net->layers[i];
        if (l.dontload) continue;
        if (l.type == CONVOLUTIONAL) {
            load_convolutional_weights_cpu(l, fp);
        }
    }
    fprintf(stderr, "Done!\n");
    fclose(fp);
}



// parser.c
convolutional_layer parse_convolutional(list *options, size_params params)
{
    int n = option_find_int(options, "filters", 1);
    int size = option_find_int(options, "size", 1);
    int stride = option_find_int(options, "stride", 1);
    int pad = option_find_int_quiet(options, "pad", 0);
    int padding = option_find_int_quiet(options, "padding", 0);
    if (pad) padding = size / 2;

    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch, h, w, c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch = params.batch;
    if (!(h && w && c)) error("Layer before convolutional layer must output image.");
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    int binary = option_find_int_quiet(options, "binary", 0);
    int xnor = option_find_int_quiet(options, "xnor", 0);
    int use_bin_output = option_find_int_quiet(options, "bin_output", 0);

    int quantized = params.quantized;
    if (params.index == 0 || activation == LINEAR || (params.index > 1 && stride>1) || size==1)
        quantized = 0; // disable Quantized for 1st and last layers
    convolutional_layer layer = make_convolutional_layer(batch, h, w, c, n, size, stride, padding, activation, batch_normalize, binary, xnor, params.net.adam, quantized, use_bin_output);
    layer.flipped = option_find_int_quiet(options, "flipped", 0);
    layer.dot = option_find_float_quiet(options, "dot", 0);
    
	if (params.net.adam) {
        layer.B1 = params.net.B1;
        layer.B2 = params.net.B2;
        layer.eps = params.net.eps;
    }

    return layer;
}

// parser.c
layer parse_region(list *options, size_params params)
{
    int coords = option_find_int(options, "coords", 4);
    int classes = option_find_int(options, "classes", 20);
    int num = option_find_int(options, "num", 1);

    layer l = make_region_layer(params.batch, params.w, params.h, num, classes, coords);
    assert(l.outputs == params.inputs);

    l.log = option_find_int_quiet(options, "log", 0);
    l.sqrt = option_find_int_quiet(options, "sqrt", 0);

    l.softmax = option_find_int(options, "softmax", 0);
    l.max_boxes = option_find_int_quiet(options, "max", 30);
    l.jitter = option_find_float(options, "jitter", .2);
    l.rescore = option_find_int_quiet(options, "rescore", 0);

    l.thresh = option_find_float(options, "thresh", .5);
    l.classfix = option_find_int_quiet(options, "classfix", 0);
    l.absolute = option_find_int_quiet(options, "absolute", 0);
    l.random = option_find_int_quiet(options, "random", 0);

    l.coord_scale = option_find_float(options, "coord_scale", 1);
    l.object_scale = option_find_float(options, "object_scale", 1);
    l.noobject_scale = option_find_float(options, "noobject_scale", 1);
    l.class_scale = option_find_float(options, "class_scale", 1);
    l.bias_match = option_find_int_quiet(options, "bias_match", 0);

    char *a = option_find_str(options, "anchors", 0);
    if (a) {
        int len = strlen(a);
        int n = 1;
        int i;
        for (i = 0; i < len; ++i) {
            if (a[i] == ',') ++n;
        }
        for (i = 0; i < n; ++i) {
            float bias = atof(a);
            l.biases[i] = bias;
            a = strchr(a, ',') + 1;
        }
    }
    return l;
}

// parser.c
softmax_layer parse_softmax(list *options, size_params params)
{
    int groups = option_find_int_quiet(options, "groups", 1);
    softmax_layer layer = make_softmax_layer(params.batch, params.inputs, groups);
    layer.temperature = option_find_float_quiet(options, "temperature", 1);
    
	//char *tree_file = option_find_str(options, "tree", 0);
    //if (tree_file) layer.softmax_tree = read_tree(tree_file);
    
	return layer;
}

// parser.c
maxpool_layer parse_maxpool(list *options, size_params params)
{
    int stride = option_find_int(options, "stride", 1);
    int size = option_find_int(options, "size", stride);
    int padding = option_find_int_quiet(options, "padding", size - 1);

    int batch, h, w, c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch = params.batch;
    if (!(h && w && c)) error("Layer before maxpool layer must output image.");

    maxpool_layer layer = make_maxpool_layer(batch, h, w, c, size, stride, padding);
    return layer;
}

// parser.c
void free_section(section *s)
{
    free(s->type);
    node *n = s->options->front;
    while (n) {
        kvp *pair = (kvp *)n->val;
        free(pair->key);
        free(pair);
        node *next = n->next;
        free(n);
        n = next;
    }
    free(s->options);
    free(s);
}

// parser.c
LAYER_TYPE string_to_layer_type(char * type)
{
    if (strcmp(type, "[region]") == 0) return REGION;
    if (strcmp(type, "[conv]") == 0
        || strcmp(type, "[convolutional]") == 0) return CONVOLUTIONAL;
    if (strcmp(type, "[net]") == 0
        || strcmp(type, "[network]") == 0) return NETWORK;
    if (strcmp(type, "[max]") == 0
        || strcmp(type, "[maxpool]") == 0) return MAXPOOL;
    if (strcmp(type, "[soft]") == 0
        || strcmp(type, "[softmax]") == 0) return SOFTMAX;
    return BLANK;
}

// parser.c
learning_rate_policy get_policy(char *s)
{
    if (strcmp(s, "random") == 0) return RANDOM;
    if (strcmp(s, "poly") == 0) return POLY;
    if (strcmp(s, "constant") == 0) return CONSTANT;
    if (strcmp(s, "step") == 0) return STEP;
    if (strcmp(s, "exp") == 0) return EXP;
    if (strcmp(s, "sigmoid") == 0) return SIG;
    if (strcmp(s, "steps") == 0) return STEPS;
    fprintf(stderr, "Couldn't find policy %s, going with constant\n", s);
    return CONSTANT;
}

// parser.c
void parse_net_options(list *options, network *net)
{
    net->batch = option_find_int(options, "batch", 1);
    net->learning_rate = option_find_float(options, "learning_rate", .001);
    net->momentum = option_find_float(options, "momentum", .9);
    net->decay = option_find_float(options, "decay", .0001);
    int subdivs = option_find_int(options, "subdivisions", 1);
    net->time_steps = option_find_int_quiet(options, "time_steps", 1);
    net->batch /= subdivs;
    net->batch *= net->time_steps;
    net->subdivisions = subdivs;

    net->adam = option_find_int_quiet(options, "adam", 0);
    if (net->adam) {
        net->B1 = option_find_float(options, "B1", .9);
        net->B2 = option_find_float(options, "B2", .999);
        net->eps = option_find_float(options, "eps", .000001);
    }

    net->h = option_find_int_quiet(options, "height", 0);
    net->w = option_find_int_quiet(options, "width", 0);
    net->c = option_find_int_quiet(options, "channels", 0);
    net->inputs = option_find_int_quiet(options, "inputs", net->h * net->w * net->c);
    net->max_crop = option_find_int_quiet(options, "max_crop", net->w * 2);
    net->min_crop = option_find_int_quiet(options, "min_crop", net->w);

    net->angle = option_find_float_quiet(options, "angle", 0);
    net->aspect = option_find_float_quiet(options, "aspect", 1);
    net->saturation = option_find_float_quiet(options, "saturation", 1);
    net->exposure = option_find_float_quiet(options, "exposure", 1);
    net->hue = option_find_float_quiet(options, "hue", 0);

    if (!net->inputs && !(net->h && net->w && net->c)) error("No input parameters supplied");

    char *policy_s = option_find_str(options, "policy", "constant");
    net->policy = get_policy(policy_s);
    net->burn_in = option_find_int_quiet(options, "burn_in", 0);
    if (net->policy == STEP) {
        net->step = option_find_int(options, "step", 1);
        net->scale = option_find_float(options, "scale", 1);
    }
    else if (net->policy == STEPS) {
        char *l = option_find(options, "steps");
        char *p = option_find(options, "scales");
        if (!l || !p) error("STEPS policy must have steps and scales in cfg file");

        int len = strlen(l);
        int n = 1;
        int i;
        for (i = 0; i < len; ++i) {
            if (l[i] == ',') ++n;
        }
        int *steps = calloc(n, sizeof(int));
        float *scales = calloc(n, sizeof(float));
        for (i = 0; i < n; ++i) {
            int step = atoi(l);
            float scale = atof(p);
            l = strchr(l, ',') + 1;
            p = strchr(p, ',') + 1;
            steps[i] = step;
            scales[i] = scale;
        }
        net->scales = scales;
        net->steps = steps;
        net->num_steps = n;
    }
    else if (net->policy == EXP) {
        net->gamma = option_find_float(options, "gamma", 1);
    }
    else if (net->policy == SIG) {
        net->gamma = option_find_float(options, "gamma", 1);
        net->step = option_find_int(options, "step", 1);
    }
    else if (net->policy == POLY || net->policy == RANDOM) {
        net->power = option_find_float(options, "power", 1);
    }
    net->max_batches = option_find_int(options, "max_batches", 0);
}

// parser.c
network parse_network_cfg(char *filename, int batch, int quantized)
{
    list *sections = read_cfg(filename);
    node *n = sections->front;
    if (!n) error("Config file has no sections");
    network net = make_network(sections->size - 1);
    net.quantized = quantized;
    net.do_input_calibration = 0;
    net.gpu_index = gpu_index;
    size_params params;
    params.quantized = quantized;

    section *s = (section *)n->val;
    list *options = s->options;
    if (strcmp(s->type, "[net]") == 0 && strcmp(s->type, "[network]") == 0)
        error("First section must be [net] or [network]");
    parse_net_options(options, &net);

    params.h = net.h;
    params.w = net.w;
    params.c = net.c;
    params.inputs = net.inputs;
    if (batch > 0) net.batch = batch;
    params.batch = net.batch;
    params.time_steps = net.time_steps;
    params.net = net;

    size_t workspace_size = 0;
    n = n->next;
    int count = 0;
    free_section(s);
    fprintf(stderr, "layer     filters    size              input                output\n");
    
	if (m_dbg) save_net_param(net);
	
	while (n) {
        params.index = count;
        fprintf(stderr, "%5d ", count);
        s = (section *)n->val;
        options = s->options;
        layer l = { 0 };
        LAYER_TYPE lt = string_to_layer_type(s->type);
        if (lt == CONVOLUTIONAL) {
            // if(count == 80) params.quantized = 0;    // doesn't lost GPU - mAP = 45.61%
            node *tmp = n->next;
            if(tmp) tmp = tmp->next;
            if (tmp)
            {
                if (string_to_layer_type(((section *)tmp->val)->type) == YOLO) {
                    params.quantized = 0;    // mAP = 53.60%
                    //printf("\n\n i = %d \n\n", count);
                }
            }

            l = parse_convolutional(options, params);
        }
        else if (lt == REGION) {
            l = parse_region(options, params);
        }
        else if (lt == YOLO) {
            //l = parse_yolo(options, params);
        }
        else if (lt == SOFTMAX) {
            l = parse_softmax(options, params);
            net.hierarchy = l.softmax_tree;
        }
        else if (lt == MAXPOOL) {
            l = parse_maxpool(options, params);
        }
        else if (lt == REORG) {
            //l = parse_reorg(options, params);
        }
        else if (lt == ROUTE) {
            //l = parse_route(options, params, net);
        }
        else if (lt == UPSAMPLE) {
            //l = parse_upsample(options, params, net);
        }
        else if (lt == SHORTCUT) {
            //l = parse_shortcut(options, params, net);
        }
        else {
            fprintf(stderr, "Type not recognized: %s\n", s->type);
        }

        l.dontload = option_find_int_quiet(options, "dontload", 0);
        l.dontloadscales = option_find_int_quiet(options, "dontloadscales", 0);
        option_unused(options);
        net.layers[count] = l;
        if (l.workspace_size > workspace_size) workspace_size = l.workspace_size;
        free_section(s);
        n = n->next;
        ++count;
        if (n) {
            params.h = l.out_h;
            params.w = l.out_w;
            params.c = l.out_c;
            params.inputs = l.outputs;
        }

		if (m_dbg) save_layer_param(l, count, lt);

    }
    free_list(sections);
    net.outputs = get_network_output_size(net);
    net.output = get_network_output(net);
    if (workspace_size) {
        //printf("%ld\n", workspace_size);
        net.workspace = calloc(1, workspace_size);
    }
    return net;
}

// -------------- gettimeofday for Windows--------------------

#if defined(_MSC_VER)
int gettimeofday(struct timeval *tv, struct timezone *tz)
{
    FILETIME ft;
    unsigned __int64 tmpres = 0;
    static int tzflag;

    if (NULL != tv)
    {
        GetSystemTimeAsFileTime(&ft);

        tmpres |= ft.dwHighDateTime;
        tmpres <<= 32;
        tmpres |= ft.dwLowDateTime;

        /*converting file time to unix epoch*/
        tmpres -= DELTA_EPOCH_IN_MICROSECS;
        tmpres /= 10;  /*convert into microseconds*/
        tv->tv_sec = (long)(tmpres / 1000000UL);
        tv->tv_usec = (long)(tmpres % 1000000UL);
    }

    if (NULL != tz)
    {
        if (!tzflag)
        {
            _tzset();
            tzflag++;
        }
        tz->tz_minuteswest = _timezone / 60;
        tz->tz_dsttime = _daylight;
    }

    return 0;
}
#endif    // _MSC_VER


float box_iou(box a, box b);

int num_detections(network *net, float thresh)
{
    int i;
    int s = 0;
    for (i = 0; i < net->n; ++i) {
        layer l = net->layers[i];
        if (l.type == DETECTION || l.type == REGION) {
            s += l.w*l.h*l.n;
        }
    }
    return s;
}

detection *make_network_boxes(network *net, float thresh, int *num)
{
    layer l = net->layers[net->n - 1];
    int i;
    int nboxes = num_detections(net, thresh);
    if (num) *num = nboxes;
    detection *dets = calloc(nboxes, sizeof(detection));
    for (i = 0; i < nboxes; ++i) {
        dets[i].prob = calloc(l.classes, sizeof(float));
        if (l.coords > 4) {
            dets[i].mask = calloc(l.coords - 4, sizeof(float));
        }
    }
    return dets;
}


void free_detections(detection *dets, int n)
{
    int i;
    for (i = 0; i < n; ++i) {
        free(dets[i].prob);
        if (dets[i].mask) free(dets[i].mask);
    }
    free(dets);
}

void find_replace(char *str, char *orig, char *rep, char *output)
{
    char buffer[4096] = { 0 };
    char *p;

    sprintf(buffer, "%s", str);
    if (!(p = strstr(buffer, orig))) {  // Is 'orig' even in 'str'?
        sprintf(output, "%s", str);
        return;
    }

    *p = '\0';

    sprintf(output, "%s%s%s", buffer, rep, p + strlen(orig));
}

void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative, int letter)
{
    int i;
    int new_w = 0;
    int new_h = 0;
    if (letter) {
        if (((float)netw / w) < ((float)neth / h)) {
            new_w = netw;
            new_h = (h * netw) / w;
        }
        else {
            new_h = neth;
            new_w = (w * neth) / h;
        }
    }
    else {
        new_w = netw;
        new_h = neth;
    }
    for (i = 0; i < n; ++i) {
        box b = dets[i].bbox;
        b.x = (b.x - (netw - new_w) / 2. / netw) / ((float)new_w / netw);
        b.y = (b.y - (neth - new_h) / 2. / neth) / ((float)new_h / neth);
        b.w *= (float)netw / new_w;
        b.h *= (float)neth / new_h;
        if (!relative) {
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}

// get prediction boxes: yolov2_forward_network.c
void get_region_boxes_cpu(layer l, int w, int h, float thresh, float **probs, box *boxes, int only_objectness, int *map);


void custom_get_region_detections(layer l, int w, int h, int net_w, int net_h, float thresh, int *map, float hier, int relative, detection *dets, int letter)
{
    box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
    float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
    int i, j;
    for (j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));
    get_region_boxes_cpu(l, 1, 1, thresh, probs, boxes, 0, map);
    for (j = 0; j < l.w*l.h*l.n; ++j) {
        dets[j].classes = l.classes;
        dets[j].bbox = boxes[j];
        dets[j].objectness = 1;
        for (i = 0; i < l.classes; ++i) {
            dets[j].prob[i] = probs[j][i];
        }
    }

    free(boxes);
    free_ptrs((void **)probs, l.w*l.h*l.n);

    //correct_region_boxes(dets, l.w*l.h*l.n, w, h, net_w, net_h, relative);
    correct_yolo_boxes(dets, l.w*l.h*l.n, w, h, net_w, net_h, relative, letter);
}

void fill_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, detection *dets, int letter)
{
    int j;
    for (j = 0; j < net->n; ++j) {
        layer l = net->layers[j];
        if (l.type == REGION) {
            custom_get_region_detections(l, w, h, net->w, net->h, thresh, map, hier, relative, dets, letter);
            //get_region_detections(l, w, h, net->w, net->h, thresh, map, hier, relative, dets);
            dets += l.w*l.h*l.n;
        }
    }
}

detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num, int letter)
{
    detection *dets = make_network_boxes(net, thresh, num);
    fill_network_boxes(net, w, h, thresh, hier, map, relative, dets, letter);
    return dets;
}


void save_convolutional_weights(layer l, int ind)
{
	char fn_conv[256], fn_norm[256];
	fprintf(stderr, "layer %d: writing weights...", ind + 1);
	sprintf(fn_conv, "dbg/conv_weights_%02d.txt", ind + 1);
	sprintf(fn_norm, "dbg/batchnorm_parameters_%02d.txt", ind + 1);
	FILE *fp_conv = fopen(fn_conv, "w");
	FILE *fp_norm = fopen(fn_norm, "w");

	for (int i = 0; i < l.n; i++) fprintf(fp_norm, "%f ", l.biases[i]);
	fprintf(fp_norm, "\n");
	if (l.batch_normalize && (!l.dontloadscales)) {
		for (int i = 0; i < l.n; i++) fprintf(fp_norm, "%f ", l.scales[i]);
		fprintf(fp_norm, "\n");
		for (int i = 0; i < l.n; i++) fprintf(fp_norm, "%f ", l.rolling_mean[i]);
		fprintf(fp_norm, "\n");
		for (int i = 0; i < l.n; i++) fprintf(fp_norm, "%f ", l.rolling_variance[i]);
		fprintf(fp_norm, "\n");
	}
	
	int num = l.n*l.c*l.size*l.size;
	if(l.xnor == 1) {
		for (int i = 0; i < num; i++) fprintf(fp_conv, "%f\n", l.binary_weights[i]);
	} else {
		for (int i = 0; i < num; i++) fprintf(fp_conv, "%f\n", l.weights[i]);
	}

	fclose(fp_conv);
	fclose(fp_norm);
	fprintf(stderr, "done.\n");
}

void save_net_param(network net)
{
	char buf[256];
	fprintf(stderr, "network init.: writing parameters...");
	sprintf(buf, "dbg/network_parameters.txt");
	FILE *fp = fopen(buf, "w");
	fprintf(fp, "m_net.quantized: %d\n", net.quantized);
	fprintf(fp, "m_net.inputs: %d\n", net.inputs);
	fprintf(fp, "m_net.h: %d\n", net.h);
	fprintf(fp, "m_net.w: %d\n", net.w);
	fprintf(fp, "m_net.c: %d\n", net.c);
	fprintf(fp, "m_net.max_crop: %d\n", net.max_crop);
	fprintf(fp, "m_net.min_crop: %d\n", net.min_crop);
	fprintf(fp, "m_net.angle: %f\n", net.angle);
	fprintf(fp, "m_net.aspect: %f\n", net.aspect);
	fprintf(fp, "m_net.saturation: %f\n", net.saturation);
	fprintf(fp, "m_net.exposure: %f\n", net.exposure);
	fprintf(fp, "m_net.hue: %f\n", net.hue);
	fprintf(fp, "m_net.learning_rate: %f\n", net.learning_rate);
	fprintf(fp, "m_net.batch: %d\n", net.batch);
	fprintf(fp, "m_net.max_batches: %d\n", net.max_batches);
	fprintf(fp, "m_net.step: %d\n", net.step);
	fprintf(fp, "m_net.scale: %f\n", net.scale);
	fprintf(fp, "m_net.num_steps: %d\n", net.num_steps);
	for (int i = 0; i < net.num_steps; i++) {
		fprintf(fp, "m_net.step[%d]      : %d\n", i, net.steps[i]);
		fprintf(fp, "m_net.scale[%d]     : %f\n", i, net.scales[i]);
	}
	fprintf(fp, "m_net.input_calibration_size: %d\n", net.input_calibration_size);
	for (int i = 0; i < net.input_calibration_size; i++) {
		fprintf(fp, "m_net.input_calibration[%d]     : %f\n", i, net.input_calibration[i]);
	}
	fprintf(fp, "m_net.time_steps: %d\n", net.time_steps);
	fprintf(fp, "m_net.burn_in: %d\n", net.burn_in);
	fprintf(fp, "m_net.policy: %d\n", net.policy);
	fprintf(fp, "m_net.gamma: %f\n", net.gamma);
	fprintf(fp, "m_net.power: %f\n", net.power);
	fprintf(fp, "m_net.decay: %f\n", net.decay);
	fprintf(fp, "m_net.adam: %d\n", net.adam);
	fprintf(fp, "m_net.B1: %f\n", net.B1);
	fprintf(fp, "m_net.B2: %f\n", net.B2);
	fprintf(fp, "m_net.eps: %f\n", net.eps);
	fprintf(fp, "m_net.burn_in: %d\n", net.burn_in);
	fprintf(fp, "m_net.momentum: %f\n", net.momentum);
	fprintf(fp, "m_net.subdivisions : %d\n", net.subdivisions);
	fclose(fp);
	fprintf(stderr, "done.\n");
}

void save_layer_param(layer l, int count, int lt)
{
	char buf[256];
	fprintf(stderr, "layer %d init.: writing parameters...", count);
	sprintf(buf, "dbg/layer_parameters_%02d_%02d.txt", count, lt);
	FILE *fp = fopen(buf, "w");
	fprintf(fp, "layer #%d (type: %d)\n", count, lt);
	fprintf(fp, "l.binary: %d\n", l.binary);
	fprintf(fp, "l.xnor: %d\n", l.xnor);
	fprintf(fp, "l.use_bin_output: %d\n", l.use_bin_output);
	fprintf(fp, "l.size: %d\n", l.size);
	fprintf(fp, "l.stride: %d\n", l.stride);
	fprintf(fp, "l.pad: %d\n", l.pad);
	fprintf(fp, "l.n (# of filters): %d\n", l.n);
	fprintf(fp, "l.batch_normalize: %d\n", l.batch_normalize);
	fprintf(fp, "l.activation(type): %d\n", l.activation);
	fprintf(fp, "l.out_h: %d\n", l.out_h);
	fprintf(fp, "l.out_w: %d\n", l.out_w);
	fprintf(fp, "l.out_c: %d\n", l.out_c);
	fprintf(fp, "l.outputs: %d\n", l.outputs);
	fclose(fp);
	fprintf(stderr, "done.\n");
}

/*
void save_layer_data(layer l, int count)
{
	char buf[256];
	fprintf(stderr, "layer %d: writing layer data...", count);
	sprintf(buf, "dbg/layer_%02d_data_%04d_%04d_%04d_%08d.txt", count, l.out_w, l.out_h, l.out_c, l.outputs);
	FILE *fp = fopen(buf, "w");
	
	if (l.out_w > 0) {
		for (int x = 0; x < l.out_w; x++)
			for (int y = 0; y < l.out_h; y++)
			{
				for (int c = 0; c < l.out_c; c++) {
					fprintf(fp, "%f", l.output[y*l.out_w*l.out_c + x*l.out_c + c]);
					if (c < (l.out_c - 1)) fprintf(fp, " ");
				}
				fprintf(fp, "\n");
			}
	}
	else {
		for (int c = 0; c < l.outputs; c++) fprintf(fp, "%f\n", l.output[c]);
	}
	
	fclose(fp);
	fprintf(stderr, "done.\n");
}
*/


void save_layer_data(layer l, int count)
{
	char buf[256];

	fprintf(stderr, "layer %d: writing layer data...", count);
	sprintf(buf, "dbg/layer_%02d_data_%04d_%04d_%04d_%08d.txt", count, l.w, l.h, l.c, l.inputs);
	FILE *fp = fopen(buf, "w");
	
	if (l.xnor) {
		if (l.w > 0) {
			for (int x = 0; x < l.w; x++)
				for (int y = 0; y < l.h; y++)
				{
					for (int c = 0; c < l.c; c++) {
						int b_i = l.binary_input[y*l.w*l.c + x*l.c + c];
						fprintf(fp, "%d", b_i);
						if (c < (l.out_c - 1)) fprintf(fp, " ");
					}
					fprintf(fp, "\n");
				}
		}
		else {
			for (int c = 0; c < l.inputs; c++) {
				int b_i = l.binary_input[c];
				fprintf(fp, "%d\n", b_i);
			}
		}
	}

	fclose(fp);
	fprintf(stderr, "done.\n");
}