#include <stdio.h>
#include <stdlib.h>

#include "box.h"

#include "Helper.h"

#include "opencv2/highgui/highgui_c.h"
#include "opencv2/core/core_c.h"
#include "opencv2/core/version.hpp"

#define OPENCV_VERSION CVAUX_STR(CV_VERSION_MAJOR)""CVAUX_STR(CV_VERSION_MINOR)""CVAUX_STR(CV_VERSION_REVISION)
#pragma comment(lib, "opencv_world" OPENCV_VERSION ".lib")

typedef struct detection_with_class {
    detection det;
    // The most probable class id: the best class index in this->prob.
    // Is filled temporary when processing results, otherwise not initialized.
    int best_class;
} detection_with_class;

// Creates array of detections with prob > thresh and fills best_class for them
detection_with_class* get_actual_detections(detection *dets, int dets_num, float thresh, int* selected_detections_num)
{
    int selected_num = 0;
    detection_with_class* result_arr = calloc(dets_num, sizeof(detection_with_class));
    int i;
    for (i = 0; i < dets_num; ++i) {
        int best_class = -1;
        float best_class_prob = thresh;
        int j;
        for (j = 0; j < dets[i].classes; ++j) {
            if (dets[i].prob[j] > best_class_prob) {
                best_class = j;
                best_class_prob = dets[i].prob[j];
            }
        }
        if (best_class >= 0) {
            result_arr[selected_num].det = dets[i];
            result_arr[selected_num].best_class = best_class;
            ++selected_num;
        }
    }
    if (selected_detections_num)
        *selected_detections_num = selected_num;
    return result_arr;
}

void draw_detections(image im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes, int ext_output)
{
    int selected_detections_num;
    detection_with_class* selected_detections = get_actual_detections(dets, num, thresh, &selected_detections_num);

    // text output
    int i;
    for (i = 0; i < selected_detections_num; ++i) {
        const int best_class = selected_detections[i].best_class;
        printf("%s: %.0f%%", names[best_class], selected_detections[i].det.prob[best_class] * 100);
        if (ext_output)
            printf("\t(left_x: %4.0f   top_y: %4.0f   width: %4.0f   height: %4.0f)\n",
            (selected_detections[i].det.bbox.x - selected_detections[i].det.bbox.w / 2)*im.w,
                (selected_detections[i].det.bbox.y - selected_detections[i].det.bbox.h / 2)*im.h,
                selected_detections[i].det.bbox.w*im.w, selected_detections[i].det.bbox.h*im.h);
        else
            printf("\n");
        int j;
        for (j = 0; j < classes; ++j) {
            if (selected_detections[i].det.prob[j] > thresh && j != best_class) {
                printf("%s: %.0f%%\n", names[j], selected_detections[i].det.prob[j] * 100);
            }
        }
    }

	if (m_dbg) {
		FILE *fp = fopen("dbg/output.txt", "w");
		fclose(fp);
	}

    // image output
    for (i = 0; i < selected_detections_num; ++i) {
        int width = im.h * .006;
        if (width < 1)
            width = 1;

        //printf("%d %s: %.0f%%\n", i, names[selected_detections[i].best_class], prob*100);
        int offset = selected_detections[i].best_class * 123457 % classes;
        float red = get_color(2, offset, classes);
        float green = get_color(1, offset, classes);
        float blue = get_color(0, offset, classes);
        float rgb[3];

        //width = prob*20+2;
        rgb[0] = red;
        rgb[1] = green;
        rgb[2] = blue;
        box b = selected_detections[i].det.bbox;
        //printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);

        int left = (b.x - b.w / 2.)*im.w;
        int right = (b.x + b.w / 2.)*im.w;
        int top = (b.y - b.h / 2.)*im.h;
        int bot = (b.y + b.h / 2.)*im.h;

        if (left < 0) left = 0;
        if (right > im.w - 1) right = im.w - 1;
        if (top < 0) top = 0;
        if (bot > im.h - 1) bot = im.h - 1;

		if (m_dbg) {
			float prob = 0.f;
			for (int j = 0; j < classes; ++j) {
				if (selected_detections[i].det.prob[j] > thresh && j == selected_detections[i].best_class) {
					prob = selected_detections[i].det.prob[j] * 100;
				}
			}
			FILE *fp = fopen("dbg/output.txt", "a+");
			fprintf(fp, "%d %.2f %d %d %d %d\n", selected_detections[i].best_class, prob, left, top, right, bot);
			fclose(fp);
		}

        draw_box_width(im, left, top, right, bot, width, red, green, blue);

    }
    free(selected_detections);
}

// --------------- Detect on the Image ---------------
// Detect on Image: this function uses other functions not from this file
void test_detector_cpu(char **names, char *cfgfile, char *weightfile, char *filename, float thresh)
{
    image **alphabet = NULL;
    network net = parse_network_cfg(cfgfile, 1);  
    
	if (weightfile) load_weights_upto_cpu(&net, weightfile, net.n);  
	else return;

    srand(2222222);
    yolov2_fuse_conv_batchnorm(net);

    calculate_binary_weights(net);

	clock_t time;
    char buff[256];
    char *input = buff;
    int j;
    float nms = .4;

	if (filename) strncpy(input, filename, 256);
    else {
		printf("Image path is not provided!"); 
		return;
    }

    image im = load_image(input, 0, 0, 3);
    image sized = resize_image(im, net.w, net.h);
    layer l = net.layers[net.n - 1];

    float *X = sized.data;
    time = clock();
        
    network_predict_cpu(net, X);

	printf("%s: Predicted in %f seconds.\n", input, (float)(clock() - time) / CLOCKS_PER_SEC); //sec(clock() - time));
        
	float hier_thresh = 0.5;
    int ext_output = 1, letterbox = 0, nboxes = 0;
        
	detection *dets = get_network_boxes(&net, im.w, im.h, thresh, hier_thresh, 1, &nboxes, letterbox);
    if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
    if(m_dbg) save_det_data_final(dets, nboxes, l.w, l.h, l.n, l.classes);

	draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes, ext_output);

    show_image(im, "predictions"); 

    free_image(im);                
    free_image(sized);             

    cvWaitKey(0);
    cvDestroyAllWindows();
}

// get command line parameters and load objects names
void run_detector(int argc, char **argv)
{
    float thresh = find_float_arg(argc, argv, "-thresh", .25);
	if (argc < 4) {
        return;
    }

    char *obj_names = argv[1];    
    char *cfg = argv[2];

	char *weights = argv[3];
	char *filename = argv[4];

    // load object names
    char **names = calloc(10000, sizeof(char *));
    int obj_count = 0;
    FILE* fp;
    char buffer[255];
    fp = fopen(obj_names, "r");
    while (fgets(buffer, 255, (FILE*)fp)) {
        names[obj_count] = calloc(strlen(buffer)+1, sizeof(char));
        strcpy(names[obj_count], buffer);
        names[obj_count][strlen(buffer) - 1] = '\0'; //remove newline
        ++obj_count;
    }

    fclose(fp);

    //if (0 == strcmp(argv[2], "test"))
	test_detector_cpu(names, cfg, weights, filename, thresh);
    
    for (int i = 0; i < obj_count; ++i) free(names[i]);
    free(names);
}

int main(int argc, char **argv)
{
    int i;
    for (i = 0; i < argc; ++i) {
        if (!argv[i]) continue;
        strip(argv[i]);
    }

	int dbg = find_float_arg(argc, argv, "-dbg", 0);

	//m_dbg = 0;
	m_dbg = dbg;

	if(m_dbg) {
		_mkdir("dbg");
	}

    if (argc < 2) {
        fprintf(stderr, "usage: %s <function>\n", argv[0]);
        return 0;
    }

    run_detector(argc, argv);

	return 0;
}
