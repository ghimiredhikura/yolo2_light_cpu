#include "additionally.h"    // some definitions from: im2col.h, blas.h, list.h, utils.h, activations.h, tree.h, layer.h, network.h
// softmax_layer.h, reorg_layer.h, route_layer.h, region_layer.h, maxpool_layer.h, convolutional_layer.h


#define W_MAX_VAL (256/2 - 1)    // 7-bit (1-bit sign)
#define I_MAX_VAL (256/2 - 1)    // 7-bit (1-bit sign)
#define R_MAX_VAL (256*256/2 - 1)    // 31-bit (1-bit sign)

#define R_MULT (32)    // 4 - 32

int max_abs(int src, int max_val)
{
    if (abs(src) > abs(max_val)) src = (src > 0) ? max_val : -max_val;
    return src;
}

short int max_abs_short(short int src, short int max_val)
{
    if (abs(src) > abs(max_val)) src = (src > 0) ? max_val : -max_val;
    return src;
}

int * get_distribution(float *arr_ptr, int arr_size, int number_of_ranges, float start_range)
{
    //const int number_of_ranges = 32;
    //const float start_range = 1.F / 65536;
    int *count = calloc(number_of_ranges, sizeof(int));
    float min_val = 10000, max_val = 0;

    int i, j;
    for (i = 0; i < arr_size; ++i) {
        float w = arr_ptr[i];

        float cur_range = start_range;
        for (j = 0; j < number_of_ranges; ++j) {
            if (fabs(cur_range) <= w && w < fabs(cur_range * 2))
                count[j]++;// , printf("found \n");
            cur_range *= 2;
            //printf("%f, ", w);
        }
    }

    return count;
}


float get_multiplier(float *arr_ptr, int arr_size, int bits_length)
{
    const int number_of_ranges = 32;
    const float start_range = 1.F / 65536;

    int i, j;
    int *count = get_distribution(arr_ptr, arr_size, number_of_ranges, start_range);

    int max_count_range = 0;
    int index_max_count = 0;
    for (j = 0; j < number_of_ranges; ++j) {
        int counter = 0;
        for (i = j; i < (j + bits_length) && i < number_of_ranges; ++i)
        {
            counter += count[i];
            //counter += log2(count[i]);
        }
        if (max_count_range < counter) {
            max_count_range = counter;
            index_max_count = j;
        }
    }
    //index_max_count = index_max_count + 2;    // optimal shift multipler
    float multiplier = 1 / (start_range * powf(2., (float)index_max_count));
    //printf(" max_count_range = %d, index_max_count = %d, multiplier = %g \n",
    //    max_count_range, index_max_count, multiplier);
    free(count);
    return multiplier;
}

// im2col.c
int8_t im2col_get_pixel_int8(int8_t *im, int height, int width, int channels,
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
void im2col_cpu_int8(int8_t* data_im,
    int channels, int height, int width,
    int ksize, int stride, int pad, int8_t* data_col)
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
                data_col[col_index] = im2col_get_pixel_int8(data_im, height, width, channels,
                    im_row, im_col, c_im, pad);
            }
        }
    }
}

// 2.9 sec
void gemm_nn_int8_int16(int M, int N, int K, int8_t ALPHA,
    int8_t *A, int lda,
    int8_t *B, int ldb,
    int16_t *C, int ldc)
{
    int32_t *c_tmp = calloc(N, sizeof(int32_t));
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k) {
            register int16_t A_PART = ALPHA*A[i*lda + k];
            //#pragma simd parallel for
            for (j = 0; j < N; ++j) {
                c_tmp[j] += A_PART*B[k*ldb + j];
                //C[i*ldc + j] += max_abs(A_PART*B[k*ldb + j] / (R_MULT), (256 * 128 - 1));
            }
        }
        for (j = 0; j < N; ++j) {
            C[i*ldc + j] += max_abs(c_tmp[j] / (R_MULT), (256 * 128 - 1));
            c_tmp[j] = 0;
        }
    }
    free(c_tmp);
}

void forward_convolutional_layer_q(layer l, network_state state)
{

    int out_h = (l.h + 2 * l.pad - l.size) / l.stride + 1;    // output_height=input_height for stride=1 and pad=1
    int out_w = (l.w + 2 * l.pad - l.size) / l.stride + 1;    // output_width=input_width for stride=1 and pad=1
    int i, f, j;
    int const out_size = out_h*out_w;
    size_t const weights_size = l.size*l.size*l.c*l.n;

    //typedef int32_t conv_t;    // l.output
    typedef int16_t conv_t;    // l.output
    conv_t *output_q = calloc(l.outputs, sizeof(conv_t));


    state.input_int8 = (int *)calloc(l.inputs, sizeof(int));
    int z;
    for (z = 0; z < l.inputs; ++z) {
        //int16_t src = lround(state.input[k] * net.layers[0].input_quant_multipler);
        int16_t src = state.input[z] * l.input_quant_multipler;
        state.input_int8[z] = max_abs(src, I_MAX_VAL);
    }
	

    // 1. Convolution !!!
    int fil;

    // cuDNN: y = conv(x)
    int m = l.n;
    int k = l.size*l.size*l.c;
    int n = out_h*out_w;
    int8_t *a = l.weights_int8;
    int8_t *b = (int8_t *)state.workspace;
    conv_t *c = output_q;    // int16_t

    // convolution as GEMM (as part of BLAS)
    //for (i = 0; i < l.batch; ++i) {
    im2col_cpu_int8(state.input_int8, l.c, l.h, l.w, l.size, l.stride, l.pad, b);    // here
    //gemm_nn_int8_int16(m, n, k, 1, a, k, b, n, c, n);    // single-thread gemm

    int t;    // multi-thread gemm
    #pragma omp parallel for
    for (t = 0; t < m; ++t) {
        gemm_nn_int8_int16(1, n, k, 1, a + t*k, k, b, n, c + t*n, n);

    }
    //}

    free(state.input_int8);

    float ALPHA1 = R_MULT / (l.input_quant_multipler * l.weights_quant_multipler);

    // cuDNN: y = alpha1 * conv(x)
    for (i = 0; i < l.outputs; ++i) {
        l.output[i] = output_q[i] * ALPHA1;    // cuDNN: alpha1
    }

    // cuDNN: y = alpha1 * conv(x) + bias
    for (fil = 0; fil < l.n; ++fil) {
        for (j = 0; j < out_size; ++j) {
            l.output[fil*out_size + j] += l.biases[fil];
        }
    }

    // cuDNN: y = act ( alpha1 * conv(x) + bias )
    // bias is always FLOAT
    if (l.activation == LEAKY) {
        for (i = 0; i < l.n*out_size; ++i) {
            l.output[i] = (l.output[i]>0) ? l.output[i] : l.output[i] / 10; //leaky_activate(l.output[i]);
        }
    }


    free(output_q);
}

#define MIN_INT8 -128

// MAX pooling layer
void forward_maxpool_layer_q(const layer l, network_state state)
{
    int b, i, j, k, m, n;
    int w_offset = -l.pad;
    int h_offset = -l.pad;

    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;

    // batch index
    for (b = 0; b < l.batch; ++b) {
        // channel index
        for (k = 0; k < c; ++k) {
            // y - input
            for (i = 0; i < h; ++i) {
                // x - input
                for (j = 0; j < w; ++j) {
                    int out_index = j + w*(i + h*(k + c*b));
                    int8_t max = MIN_INT8;
                    int max_i = -1;
                    // pooling x-index
                    for (n = 0; n < l.size; ++n) {
                        // pooling y-index
                        for (m = 0; m < l.size; ++m) {
                            int cur_h = h_offset + i*l.stride + n;
                            int cur_w = w_offset + j*l.stride + m;
                            int index = cur_w + l.w*(cur_h + l.h*(k + b*l.c));
                            int valid = (cur_h >= 0 && cur_h < l.h &&
                                cur_w >= 0 && cur_w < l.w);
                            int8_t val = (valid != 0) ? state.input_int8[index] : MIN_INT8;
                            max_i = (val > max) ? index : max_i;    // get max index
                            max = (val > max) ? val : max;            // get max value
                        }
                    }
                    //l.output[out_index] = max;        // store max value
                    l.output_int8[out_index] = max;        // store max value
                    l.indexes[out_index] = max_i;    // store max index
                }
            }
        }
    }
}


// Route layer - just copy 1 or more layers into the current layer
void forward_route_layer_q(const layer l, network_state state)
{
    int i, j;
    int offset = 0;
    // number of merged layers
    for (i = 0; i < l.n; ++i) {
        int index = l.input_layers[i];                    // source layer index
                                                        //float *input = state.net.layers[index].output;    // source layer output ptr
        int8_t *input = state.net.layers[index].output_int8;    // source layer output ptr
        int input_size = l.input_sizes[i];                // source layer size
                                                        // batch index
        for (j = 0; j < l.batch; ++j) {
            memcpy(l.output_int8 + offset + j*l.outputs, input + j*input_size, input_size * sizeof(int8_t));
        }
        offset += input_size;
    }
}

// ---- region layer ----

static void softmax_q(float *input, int n, float temp, float *output)
{
    int i;
    float sum = 0;
    float largest = -FLT_MAX;
    for (i = 0; i < n; ++i) {
        if (input[i] > largest) largest = input[i];
    }
    for (i = 0; i < n; ++i) {
        float e = expf(input[i] / temp - largest / temp);
        sum += e;
        output[i] = e;
    }
    for (i = 0; i < n; ++i) {
        output[i] /= sum;
    }
}

// Region layer - just change places of array items, then do logistic_activate and softmax
void forward_region_layer_q(const layer l, network_state state)
{
    int i, b;
    int size = l.coords + l.classes + 1;    // 4 Coords(x,y,w,h) + Classes + 1 Probability-t0
                                            //printf("\n l.coords = %d \n", l.coords);
    memcpy(l.output, state.input, l.outputs*l.batch * sizeof(float));

    //flatten(l.output, l.w*l.h, size*l.n, l.batch, 1);
    // convert many channels to the one channel (depth=1)
    // (each grid cell will have a number of float-variables equal = to the initial number of channels)
    {
        float *x = l.output;
        int layer_size = l.w*l.h;    // W x H - size of layer
        int layers = size*l.n;        // number of channels (where l.n = number of anchors)
        int batch = l.batch;

        float *swap = calloc(layer_size*layers*batch, sizeof(float));
        int i, c, b;
        // batch index
        for (b = 0; b < batch; ++b) {
            // channel index
            for (c = 0; c < layers; ++c) {
                // layer grid index
                for (i = 0; i < layer_size; ++i) {
                    int i1 = b*layers*layer_size + c*layer_size + i;
                    int i2 = b*layers*layer_size + i*layers + c;
                    swap[i2] = x[i1];
                }
            }
        }
        memcpy(x, swap, layer_size*layers*batch * sizeof(float));
        free(swap);
    }


    // logistic activation only for: t0 (where is t0 = Probability * IoU(box, object))
    for (b = 0; b < l.batch; ++b) {
        // for each item (x, y, anchor-index)
        for (i = 0; i < l.h*l.w*l.n; ++i) {
            int index = size*i + b*l.outputs;
            float x = l.output[index + 4];
            l.output[index + 4] = 1.0F / (1.0F + expf(-x));    // logistic_activate_q(l.output[index + 4]);
        }
    }

    //else if (l.softmax) 
	{    // Yolo v2
                            // softmax activation only for Classes probability
        for (b = 0; b < l.batch; ++b) {
            // for each item (x, y, anchor-index)
            //#pragma omp parallel for
            for (i = 0; i < l.h*l.w*l.n; ++i) {
                int index = size*i + b*l.outputs;
                softmax_q(l.output + index + 5, l.classes, 1, l.output + index + 5);
            }
        }
    }

}

void yolov2_forward_network_q(network net, network_state state)
{
    state.workspace = net.workspace;
    int i, k;
    for (i = 0; i < net.n; ++i) {
        state.index = i;
        layer l = net.layers[i];

        if (l.type == CONVOLUTIONAL) {
            if (i >= 1 && l.activation != LINEAR) forward_convolutional_layer_q(l, state);
            else forward_convolutional_layer_cpu(l, state);

            printf("\n %d - CONVOLUTIONAL \t\t l.size = %d  \n", i, l.size);
        }
        else if (l.type == MAXPOOL) {
            forward_maxpool_layer_cpu(l, state);
            //printf("\n MAXPOOL \t\t l.size = %d  \n", l.size);
        }
        else if (l.type == ROUTE) {
            //forward_route_layer_cpu(l, state);
            //printf("\n ROUTE \t\t\t l.n = %d  \n", l.n);
        }
        else if (l.type == REORG) {
            //forward_reorg_layer_cpu(l, state);
            //printf("\n REORG \n");
        }
        else if (l.type == UPSAMPLE) {
            //forward_upsample_layer_cpu(l, state);
            //printf("\n UPSAMPLE \n");
        }
        else if (l.type == SHORTCUT) {
            //forward_shortcut_layer_cpu(l, state);
            //printf("\n SHORTCUT \n");
        }
        else if (l.type == YOLO) {
            //forward_yolo_layer_cpu(l, state);
            //printf("\n YOLO \n");
        }
        else if (l.type == REGION) {
            forward_region_layer_cpu(l, state);
            //printf("\n REGION \n");
        }
        else {
            printf("\n layer: %d \n", l.type);
        }


        state.input = l.output;
        //state.input_int8 = l.output_int8;
    }
}

// detect on CPU
float *network_predict_quantized(network net, float *input)
{
    network_state state;
    state.net = net;
    state.index = 0;
    state.input = input;
    //state.input_int8 = calloc(net.w*net.h*net.c, sizeof(int8_t));
    state.truth = 0;
    state.train = 0;
    state.delta = 0;

    yolov2_forward_network_q(net, state);    // network on CPU
                                            //float *out = get_network_output(net);
    int i;
    for (i = net.n - 1; i > 0; --i) if (net.layers[i].type != COST) break;
    //free(state.input_int8);
    return net.layers[i].output;
}


// --------------------
// x - last conv-layer output
// biases - anchors from cfg-file
// n - number of anchors from cfg-file
box get_region_box_q(float *x, float *biases, int n, int index, int i, int j, int w, int h)
{
    box b;
    b.x = (i + logistic_activate(x[index + 0])) / w;    // (col + 1./(1. + exp(-x))) / width_last_layer
    b.y = (j + logistic_activate(x[index + 1])) / h;    // (row + 1./(1. + exp(-x))) / height_last_layer
    b.w = expf(x[index + 2]) * biases[2 * n] / w;        // exp(x) * anchor_w / width_last_layer
    b.h = expf(x[index + 3]) * biases[2 * n + 1] / h;    // exp(x) * anchor_h / height_last_layer
    return b;
}

// get prediction boxes
void get_region_boxes_q(layer l, int w, int h, float thresh, float **probs, box *boxes, int only_objectness, int *map)
{
    int i, j, n;
    float *predictions = l.output;
    // grid index
    for (i = 0; i < l.w*l.h; ++i) {
        int row = i / l.w;
        int col = i % l.w;
        // anchor index
        for (n = 0; n < l.n; ++n) {
            int index = i*l.n + n;    // index for each grid-cell & anchor
            int p_index = index * (l.classes + 5) + 4;
            float scale = predictions[p_index];                // scale = t0 = Probability * IoU(box, object)
            if (l.classfix == -1 && scale < .5) scale = 0;    // if(t0 < 0.5) t0 = 0;
            int box_index = index * (l.classes + 5);
            boxes[index] = get_region_box_q(predictions, l.biases, n, box_index, col, row, l.w, l.h);
            boxes[index].x *= w;
            boxes[index].y *= h;
            boxes[index].w *= w;
            boxes[index].h *= h;

            int class_index = index * (l.classes + 5) + 5;          
			{
                // Yolo v2
                for (j = 0; j < l.classes; ++j) {
                    float prob = scale*predictions[class_index + j];    // prob = IoU(box, object) = t0 * class-probability
                    probs[index][j] = (prob > thresh) ? prob : 0;        // if (IoU < threshold) IoU = 0;
                }
            }
            if (only_objectness) {
                probs[index][0] = scale;
            }
        }
    }
}


// Quantinization and get multiplers for convolutional weights for quantinization
void quantinization_and_get_multipliers(network net)
{
    // ----------- entropy_calibration(,, 1.0 / 16, 4096); - FULL ----------------------
    //float input_mult[] = { 256, 4,32,64,32,32,32,32,32,64,64,64,64,64,128,64,128,128,64,128,64,128,128 };    // divided 4 - full works
    int counter = 0;
    //const int input_mult_size = sizeof(input_mult) / sizeof(float);

    int j;
    for (j = 0; j < net.n; ++j) {
        layer *l = &net.layers[j];

        if (l->type == CONVOLUTIONAL) {
            size_t const weights_size = l->size*l->size*l->c*l->n;
            size_t const filter_size = l->size*l->size*l->c;

            int i, k, fil;

            // get optimal multipliers - for Weights
            //float *weights_multiplier = (float *)calloc(l->n, sizeof(float));
            //l->output_multipler = (float *)calloc(l->n, sizeof(float));

            //float weights_multiplier_single = entropy_calibration(l->weights, weights_size, 1.0 / (2048), (2048));

            //float weights_multiplier_single = entropy_calibration(l->weights, weights_size, 1.0 / 4096, 4096) / 2;
            //if (j == 0) weights_multiplier_single = entropy_calibration(l->weights, weights_size, 1.0 / 2, 2048);

            float old_weight_mult = get_multiplier(l->weights, weights_size, 8) / 4;    // good [2 - 8], best 4
            float weights_multiplier_single = old_weight_mult;

            //float old_weight_mult = get_multiplier(l->weights, weights_size, 7) / 4;
            printf(" old_weight_mult = %f, weights_multiplier_single = %f \n\n", old_weight_mult, weights_multiplier_single);
            //weights_multiplier_single = old_weight_mult;


            l->weights_quant_multipler = weights_multiplier_single;


            for (fil = 0; fil < l->n; ++fil) {
                for (i = 0; i < filter_size; ++i) {
                    float w = l->weights[fil*filter_size + i] * l->weights_quant_multipler;// [fil];
                    l->weights_int8[fil*filter_size + i] = max_abs(w, W_MAX_VAL);
                    //l->weights_int8[fil*filter_size + i] = max_abs(lround(w), W_MAX_VAL);
                }
            }


            if (counter >= net.input_calibration_size) {
                printf("\n Warning: input_calibration= in the cfg-file has less values %d than convolutional layers %d \n",
                    net.input_calibration_size, counter);
            }

            //l->input_quant_multipler = 40;//(counter < net.input_calibration_size) ? net.input_calibration[counter] : 16;    // best 40
            l->input_quant_multipler = (counter < net.input_calibration_size) ? net.input_calibration[counter] : 40;


            ++counter;

            //float current_input_mult = 40;//(counter < net.input_calibration_size) ? net.input_calibration[counter] : 16;
            float current_input_mult = (counter < net.input_calibration_size) ? net.input_calibration[counter] : 40;


            for (fil = 0; fil < l->n; ++fil) {
                if (counter == 1) l->output_multipler = current_input_mult / (l->weights_quant_multipler * l->input_quant_multipler / R_MULT);
                if (counter == 2) l->output_multipler = current_input_mult / (l->weights_quant_multipler * l->input_quant_multipler / R_MULT);
                else if (counter >= 2) l->output_multipler = current_input_mult / (l->weights_quant_multipler * l->input_quant_multipler / R_MULT);
            }


            // quantinization Biases
            for (fil = 0; fil < l->n; ++fil) {
                // calculate optimal multipliers - for Biases
                float biases_multipler = (l->output_multipler * l->weights_quant_multipler * l->input_quant_multipler / R_MULT);

                l->biases_quant[fil] = l->biases[fil] * biases_multipler;
            }

            printf(" Multiplers: weights %g, input %g, output %g \n",
                l->weights_quant_multipler, l->input_quant_multipler, l->output_multipler);
        }
        else {
            printf(" Skip layer: %d \n", l->type);
        }
    }
}
