/***********************
 * Add two double vectors on a GPU
 * The program is an extened version of an example program 
 * without error checking from Nvidia Corporation, 
 * written by Jens Graebel and Peer Ueberholz
 ***********************/

//#define CL_TARGET_OPENCL_VERSION 220

#include <stdio.h>
#include <CL/cl.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <ctime>

// OpenCL variables

/***********************
 * kernel for adding two vectors
 * kernel will be compiled during runtime
 ***********************/
const char *sProgramSource[]={
        "__kernel void vectorAdd(__global const float *a,",
        "                        __global const float *b,",
        "                        __global float *c)",
        "{",
        "   int i=get_global_id(0);",
        "   c[i]=a[i]+b[i];",
        "}"
};

//int row = blockIdx.y * blockDim.y + threadIdx.y
const char *sMatrixMultiplication[]={
        "__kernel void matrix_matrix_multiplication(__global const float *a,",
        "                        __global const float *b,",
        "                        __global float *c, int n)",
        "{",
        "",
        "   int row = get_group_id(1) * get_local_size(1) + get_local_id(1);",
        "   int col = get_group_id(0) * get_local_size(0) + get_local_id(0);",
        "   ",
        "   int tempSum = 0;",
        "   if( (row < n) && (col < n) ) {",
        "       for(int k = 0; k < n; k++) {",
        "               ",
        "               tempSum += a[row * n + k] * b[k * n + col];",
        "       }",
        "       c[row * n + col] = tempSum;",
        "   }",
        "}"
};

double second()
{
    struct timeval tm;
    double t ;
    gettimeofday(&tm,NULL);
    t = (double) (tm.tv_sec) + ((double) (tm.tv_usec))/1.0e6;
    return t;
}

/***********************
 * create OpenCL platform, device, context & queue
 ***********************/
void init_OpenCL(cl_context * hContext, cl_command_queue * hCmdQueue,
                 cl_device_id * devices) {

    cl_int err;             //errorFlag

    // get platform ID, print general information of the OpenCL-platform
    cl_platform_id platforms[100];
    cl_uint platforms_n;
    err = clGetPlatformIDs(100,platforms, &platforms_n);
    if (err != CL_SUCCESS)
        printf("error getting platform IDS\n");
    for (unsigned int i=0; i<platforms_n; i++)
    {
        char buffer[10240];
        printf("  -- %d --\n", i);
        clGetPlatformInfo(platforms[i],CL_PLATFORM_PROFILE,10240, buffer,NULL);
        printf("  PROFILE = %s\n", buffer);
        clGetPlatformInfo(platforms[i],CL_PLATFORM_VERSION,10240, buffer,NULL);
        printf("  VERSION = %s\n", buffer);
        clGetPlatformInfo(platforms[i],CL_PLATFORM_NAME,10240, buffer,NULL);
        printf("  NAME = %s\n", buffer);
        clGetPlatformInfo(platforms[i],CL_PLATFORM_VENDOR,10240, buffer,NULL);
        printf("  VENDOR = %s\n", buffer);
        clGetPlatformInfo(platforms[i],CL_PLATFORM_EXTENSIONS,10240,buffer,NULL);
        printf("  EXTENSIONS = %s\n", buffer);
    }

    cl_uint devices_n;
    // get the GPU-devices of platform 0, print details of the device
    err = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_GPU, 100, devices,
                          &devices_n);
    if (err != CL_SUCCESS) {
        printf("error getting device IDS\n");
        if(err == CL_INVALID_PLATFORM)
            printf("clGetDeviceIDs CL_INVALID_PLATFORM\n");
        else if(err == CL_INVALID_DEVICE_TYPE)
            printf("clGetDeviceIDs CL_INVALID_DEVICE_TYPE\n");
        else if(err == CL_INVALID_VALUE)
            printf("clGetDeviceIDs CL_INVALID_VALUE\n");
        else if(err == CL_DEVICE_NOT_FOUND)
            printf("clGetDeviceIDs CL_DEVICE_NOT_FOUND\n");
    }

    printf("=== %d OpenCL device(s) found on platform: %d\n\n", platforms_n,
           devices_n);
    for (unsigned int i=0; i<devices_n; i++)
    {
        char buffer[10240];
        cl_uint buf_uint;
        cl_ulong buf_ulong;
        printf("  -- %d --\n", i);
        (clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(buffer),
                         buffer, NULL));
        printf("  DEVICE_NAME = %s\n", buffer);
        (clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, sizeof(buffer),
                         buffer, NULL));
        printf("  DEVICE_VENDOR = %s\n", buffer);
        (clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, sizeof(buffer),
                         buffer, NULL));
        printf("  DEVICE_VERSION = %s\n", buffer);
        (clGetDeviceInfo(devices[i], CL_DRIVER_VERSION, sizeof(buffer),
                         buffer, NULL));
        printf("  DRIVER_VERSION = %s\n", buffer);
        (clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS,
                         sizeof(buf_uint), &buf_uint, NULL));
        printf("  DEVICE_MAX_COMPUTE_UNITS = %u\n", (unsigned int)buf_uint);
        (clGetDeviceInfo(devices[i], CL_DEVICE_MAX_CLOCK_FREQUENCY,
                         sizeof(buf_uint), &buf_uint, NULL));
        printf("  DEVICE_MAX_CLOCK_FREQUENCY = %u\n", (unsigned int)buf_uint);
        (clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE,
                         sizeof(buf_ulong), &buf_ulong, NULL));
        printf("  DEVICE_GLOBAL_MEM_SIZE = %u\n\n", (unsigned int)buf_ulong);
    }
    if (devices_n == 0)
    {
        printf("error, on platform 0, there is no GPU device\n");
        exit(1);
    }

    // set the context properties of the GPU devices of platform 0
    // and create a context for all devices of type CL_DEVICE_TYPE_GPU
    // on platform 0
    //
    // An alternitive to this method would be to create an context
    // directly for one device with clCreateContext
    cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM,
                                    (cl_context_properties)platforms[0],0};
    *hContext = clCreateContextFromType(cps, CL_DEVICE_TYPE_GPU, 0, 0, &err);

    // query all devices available to the context
    // instead of this one can use devices[i] directly
    size_t nContextDescriptorSize;
    clGetContextInfo(*hContext, CL_CONTEXT_DEVICES, 0, 0, &nContextDescriptorSize);
    cl_device_id * aDevices = (cl_device_id*)malloc(nContextDescriptorSize);
    clGetContextInfo(*hContext, CL_CONTEXT_DEVICES, nContextDescriptorSize,
                     aDevices, 0);
    // create a command queue for first device the context reported
    *hCmdQueue = clCreateCommandQueue(*hContext, aDevices[0], 0, &err);
    if (err != CL_SUCCESS)
        printf("error clCreateCommandQueue\n");

}

/***********************
 * create a kernel out of string
 ***********************/
void create_kernel(cl_kernel * hKernel, cl_context hContext, cl_device_id* devices) {

    cl_int err;             //errorFlag

    // create & compile kernel program out of the char arrax sProgramSource
    cl_program hProgram;
    cl_uint count = sizeof(sProgramSource)/sizeof(*sProgramSource);
    hProgram = clCreateProgramWithSource(hContext,count,sProgramSource,0,&err);
    if (err != CL_SUCCESS)
        printf("error clCreateProgramWithSource\n");
    err = clBuildProgram(hProgram, 0, 0, 0, 0, 0);
    if (err != CL_SUCCESS) {
        printf("error clBuildProgram\n");
        size_t logSize;
        clGetProgramBuildInfo(hProgram, devices[0], CL_PROGRAM_BUILD_LOG, 0, 0, &logSize);
        char log[logSize];
        clGetProgramBuildInfo(hProgram, devices[0], CL_PROGRAM_BUILD_LOG, logSize, (void*)log, NULL);
        printf(log);
        exit(1);
    }

    // create kernel
    *hKernel = clCreateKernel(hProgram, "vectorAdd", 0);

    clReleaseProgram(hProgram);
}

/***********************
 * create a kernel out of string
 ***********************/
void create_kernel_matrix_multiplication(cl_kernel * hKernel, cl_context hContext, cl_device_id* devices) {

    cl_int err;             //errorFlag

    // create & compile kernel program out of the char arrax sMatrixMultiplication
    cl_program hProgram;
    cl_uint count = sizeof(sMatrixMultiplication)/sizeof(*sMatrixMultiplication);
    hProgram = clCreateProgramWithSource(hContext,count,sMatrixMultiplication,0,&err);
    if (err != CL_SUCCESS)
        printf("error clCreateProgramWithSource\n");
    err = clBuildProgram(hProgram, 0, 0, 0, 0, 0);
    if (err != CL_SUCCESS) {
        printf("error clBuildProgram\n");
        size_t logSize;
        clGetProgramBuildInfo(hProgram, devices[0], CL_PROGRAM_BUILD_LOG, 0, 0, &logSize);
        char log[logSize];
        clGetProgramBuildInfo(hProgram, devices[0], CL_PROGRAM_BUILD_LOG, logSize, (void*)log, NULL);
        printf(log);
        exit(1);
    }

    // create kernel
    *hKernel = clCreateKernel(hProgram, "matrix_matrix_multiplication", 0);

    clReleaseProgram(hProgram);
}

/***********************
 *  allocate memory, set the parameters for the kernel, run kernel and clean up
 ***********************/
void run_kernel(cl_kernel hKernel, cl_context hContext,
                cl_command_queue hCmdQueue) {

    size_t local_size=256;
    size_t group_size=16;
    const size_t cnDimension=local_size*group_size;

    cl_int err;             //errorFlag

    // allocate host vectors
    float * pA = new  float[cnDimension];
    float * pB = new  float[cnDimension];
    float * pC = new  float[cnDimension];

    // initialize host memory
    for (cl_uint ii=0; ii< cnDimension; ii++) {
        pA[ii] = (float)rand()/(float) (RAND_MAX)+2.0;
        pB[ii] = (float)rand()/(float) (RAND_MAX)+2.0;
        pC[ii] = 0.0;
    }

    // allocate device memory
    cl_mem hDeviceMemA, hDeviceMemB, hDeviceMemC;
    hDeviceMemA = clCreateBuffer(hContext,
                                 CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 cnDimension * sizeof(cl_float), pA, &err);
    if (err != CL_SUCCESS)
        printf("error clCreateBuffer\n");

    hDeviceMemB = clCreateBuffer(hContext,
                                 CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 cnDimension * sizeof(cl_float), pB, &err);
    if (err != CL_SUCCESS)
        printf("error clCreateBuffer\n");

    hDeviceMemC = clCreateBuffer(hContext,
                                 CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
                                 cnDimension * sizeof(cl_float), pC, &err);
    if (err != CL_SUCCESS)
        printf("error clCreateBuffer\n");


    // setup parameter values
    clSetKernelArg(hKernel, 0, sizeof(cl_mem), (void *)&hDeviceMemA);
    clSetKernelArg(hKernel, 1, sizeof(cl_mem), (void *)&hDeviceMemB);
    clSetKernelArg(hKernel, 2, sizeof(cl_mem), (void *)&hDeviceMemC);

    // execute kernel
    //    cl_int clEnqueueNDRangeKernel (cl_command_queue command_queue,
    //	cl_kernel kernel,
    //	cl_uint work_dim,                   (hier 1)
    //	const size_t *global_work_offset,   (kein Offset)
    //	const size_t *global_work_size,     (cndDimension)
    //	const size_t *local_work_size,      (local_size)
    //	cl_uint num_events_in_wait_list,    (0)
    //	const cl_event *event_wait_list,    (0)
    //	cl_event *event)
    cl_event tester;
    clEnqueueNDRangeKernel(hCmdQueue, hKernel, 1, 0, &cnDimension,
                           &local_size, 0, NULL, &tester);
    clWaitForEvents(1, &tester);

    // copy results from device back to host.
    // Be careful, without waiting for the event,
    // the kernes is executed asynchronous,
    // but this command blocks until the job is done
    err = clEnqueueReadBuffer(hCmdQueue, hDeviceMemC, CL_TRUE, 0,
                              cnDimension * sizeof(cl_float), pC, 0, NULL, NULL);
    if (err != CL_SUCCESS)
        printf("error clEnqueueReadBuffer\n");

    printf("program ready with %10.4e (pA[3]) +  %10.4e (pB[3]) = %10.4e\n",
           pA[3],pB[3],pC[3]);


    // now clean everything
    delete[] pA;
    delete[] pB;
    delete[] pC;

    clReleaseMemObject(hDeviceMemA);
    clReleaseMemObject(hDeviceMemB);
    clReleaseMemObject(hDeviceMemC);
    clReleaseKernel(hKernel);
    clReleaseCommandQueue(hCmdQueue);
    clReleaseContext(hContext);
}

void run_kernel_matrix_multiplication(cl_kernel hKernel, cl_context hContext, cl_command_queue hCmdQueue) {

    size_t matrix_size = 1024;

    size_t local_size = 16;

    size_t cnDimension = matrix_size * matrix_size;

    cl_int err;             //errorFlag

    // allocate host matrices
    float ** pA = (float **) malloc(matrix_size * sizeof(float*));
    float ** pB = (float **) malloc(matrix_size * sizeof(float*));
    float ** pC = (float **) malloc(matrix_size * sizeof(float*));

    pA[0] = (float *) malloc(matrix_size * matrix_size * sizeof (float ));
    pB[0] = (float *) malloc(matrix_size * matrix_size * sizeof (float ));
    pC[0] = (float *) malloc(matrix_size * matrix_size * sizeof (float ));

    for(cl_uint i = 1; i < matrix_size; i++) {

        pA[i] = pA[i - 1] + matrix_size;
        pB[i] = pB[i - 1] + matrix_size;
        pC[i] = pC[i - 1] + matrix_size;
    }

    // initialize host memory
    for (cl_uint row=0; row < matrix_size; row++) {

        for(cl_int col = 0; col < matrix_size; col++) {

            pA[row][col] = (float)rand() / (float) (RAND_MAX) + 2.0;
            pB[row][col] = (float)rand() / (float) (RAND_MAX) + 2.0;
            pC[row][col] = 0;
        }
    }

    // allocate device memory
    cl_mem hDeviceMemA, hDeviceMemB, hDeviceMemC;
    hDeviceMemA = clCreateBuffer(hContext,
                                 CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 cnDimension * sizeof(cl_float), pA[0], &err);
    if (err != CL_SUCCESS)
        printf("error clCreateBuffer\n");

    hDeviceMemB = clCreateBuffer(hContext,
                                 CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 cnDimension * sizeof(cl_float), pB[0], &err);
    if (err != CL_SUCCESS)
        printf("error clCreateBuffer\n");

    hDeviceMemC = clCreateBuffer(hContext,
                                 CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
                                 cnDimension * sizeof(cl_float), pC[0], &err);
    if (err != CL_SUCCESS)
        printf("error clCreateBuffer\n");


    // setup parameter values
    clSetKernelArg(hKernel, 0, sizeof(cl_mem), (void *)&hDeviceMemA);
    clSetKernelArg(hKernel, 1, sizeof(cl_mem), (void *)&hDeviceMemB);
    clSetKernelArg(hKernel, 2, sizeof(cl_mem), (void *)&hDeviceMemC);
    clSetKernelArg(hKernel, 3, sizeof(size_t), (void *)&matrix_size);

    // execute kernel
    //    cl_int clEnqueueNDRangeKernel (cl_command_queue command_queue,
    //	cl_kernel kernel,
    //	cl_uint work_dim,                   (hier 1)
    //	const size_t *global_work_offset,   (kein Offset)
    //	const size_t *global_work_size,     (cndDimension)
    //	const size_t *local_work_size,      (local_size)
    //	cl_uint num_events_in_wait_list,    (0)
    //	const cl_event *event_wait_list,    (0)
    //	cl_event *event)
    cl_event tester;

    clEnqueueNDRangeKernel(hCmdQueue, hKernel, 2, 0, &cnDimension,
                           &local_size, 0, NULL, &tester);
    clWaitForEvents(1, &tester);

    // copy results from device back to host.
    // Be careful, without waiting for the event,
    // the kernes is executed asynchronous,
    // but this command blocks until the job is done
    err = clEnqueueReadBuffer(hCmdQueue, hDeviceMemC, CL_TRUE, 0,
                              cnDimension * sizeof(cl_float), pC, 0, NULL, NULL);
    if (err != CL_SUCCESS)
        printf("error clEnqueueReadBuffer\n");

    printf("matrix matrix multiplication ready with %10.4e (pA[3][3]) ;  %10.4e (pB[3][3]) ; %10.4e (pC[3][3])\n",
           pA[3][3],pB[3][3],pC[3][3]);

    // now clean everything
    free(pA[0]);
    free(pA);
    free(pB[0]);
    free(pB);
    free(pC[0]);
    free(pC);


    clReleaseMemObject(hDeviceMemA);
    clReleaseMemObject(hDeviceMemB);
    clReleaseMemObject(hDeviceMemC);
    clReleaseKernel(hKernel);
    clReleaseCommandQueue(hCmdQueue);
    clReleaseContext(hContext);
}

/***********************
 * main function
 ***********************/
int main() {

    // create OpenCL context and queue
    cl_context hContext;
    cl_command_queue hCmdQueue;
    cl_device_id devices[100];
    init_OpenCL(&hContext, &hCmdQueue, devices);

    // now everything is prepared to create an executable source code
    cl_kernel hKernel;
    //create_kernel(&hKernel, hContext, devices);
    create_kernel_matrix_multiplication(&hKernel, hContext, devices);

    // allocate memory, set the parameters for the kernel, run kernel and clean up
    //run_kernel(hKernel, hContext, hCmdQueue);
    run_kernel_matrix_multiplication(hKernel, hContext, hCmdQueue);


    return 0;
}