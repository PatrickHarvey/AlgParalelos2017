#include <stdio.h>
#include <stdlib.h>

#include "kmeans.h"

static inline int nextPowerOfTwo(int n) {
    n--;

    n = n >>  1 | n;
    n = n >>  2 | n;
    n = n >>  4 | n;
    n = n >>  8 | n;
    n = n >> 16 | n;
//  n = n >> 32 | n;    //  64-bits

    return ++n;
}

//distancia euclidea
__host__ __device__ inline static
float euclid_dist_2(int    numCoords,
                    int    numObjs,
                    int    numClusters,
                    float *objects,     // [numCoords][numObjs]
                    float *clusters,    // [numCoords][numClusters]
                    int    objectId,
                    int    clusterId)
{
    int i;
    float ans=0.0;

    for (i = 0; i < numCoords; i++) {
        ans += (objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]) *
               (objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]);
    }

    return(ans);
}

// encuentra el cluster más cercano
__global__ static
void find_nearest_cluster(int numCoords,
                          int numObjs,
                          int numClusters,
                          float *objects,           //  [numCoords][numObjs]
                          float *deviceClusters,    //  [numCoords][numClusters]
                          int *membership,          //  [numObjs]
                          int *intermediates)
{
    extern __shared__ char sharedMemory[];

    // Hay elementos blockDim.x, uno para cada hilo en el bloque
    unsigned char *membershipChanged = (unsigned char *)sharedMemory;
#if BLOCK_SHARED_MEM_OPTIMIZATION
    float *clusters = (float *)(sharedMemory + blockDim.x);
#else
    float *clusters = deviceClusters;
#endif

    membershipChanged[threadIdx.x] = 0;

#if BLOCK_SHARED_MEM_OPTIMIZATION

    //  Se puede sobrecargar la memoria compartida si hay muchos clusters o muchas coordenadas

    for (int i = threadIdx.x; i < numClusters; i += blockDim.x) {
        for (int j = 0; j < numCoords; j++) {
            clusters[numClusters * j + i] = deviceClusters[numClusters * j + i];
        }
    }
    __syncthreads();
#endif

    int objectId = blockDim.x * blockIdx.x + threadIdx.x;

    if (objectId < numObjs) {
        int   index, i;
        float dist, min_dist;

        /* encuentra el id del cluster más cercano */
        index    = 0;
        min_dist = euclid_dist_2(numCoords, numObjs, numClusters, objects, clusters, objectId, 0);

        for (i=1; i<numClusters; i++)
	{
            dist = euclid_dist_2(numCoords, numObjs, numClusters, objects, clusters, objectId, i);
            if (dist < min_dist)
	    {
                min_dist = dist;
                index    = i;
            }
        }

        if (membership[objectId] != index) {
            membershipChanged[threadIdx.x] = 1;
        }

        /* asigna el id al objeto */
        membership[objectId] = index;

        __syncthreads(); 

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                membershipChanged[threadIdx.x] +=
                    membershipChanged[threadIdx.x + s];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            intermediates[blockIdx.x] = membershipChanged[0];
        }
    }
}

__global__ static
void compute_delta(int *deviceIntermediates, int numIntermediates, int numIntermediates2)  
{

    extern __shared__ unsigned int intermediates[];

    intermediates[threadIdx.x] =
        (threadIdx.x < numIntermediates) ? deviceIntermediates[threadIdx.x] : 0;

    __syncthreads();

    for (unsigned int s = numIntermediates2 / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            intermediates[threadIdx.x] += intermediates[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        deviceIntermediates[0] = intermediates[0];
    }
}

//  DATA LAYOUT
//
//  objects         [numObjs][numCoords]
//  clusters        [numClusters][numCoords]
//  dimObjects      [numCoords][numObjs]
//  dimClusters     [numCoords][numClusters]
//  newClusters     [numCoords][numClusters]
//  deviceObjects   [numCoords][numObjs]
//  deviceClusters  [numCoords][numClusters]
//  ----------------------------------------
//
/* retorna un array de centro de clusters de tamaño [numClusters][numCoords]       */
float** cuda_kmeans(float **objects,      /* entrada: [numObjs][numCoords] */
                   int     numCoords,    /* num caracteristicas */
                   int     numObjs,      /* num objetos */
                   int     numClusters,  /* num clusters */
                   float   threshold, 	
                   int    *membership,   /* salida: [numObjs] */
                   int    *loop_iterations)
{
    int      i, j, index, loop=0;
    int     *newClusterSize; /* [numClusters]: num de objetos nuevos en el cluster */
    float    delta;          
    float  **dimObjects;
    float  **clusters;       /* salida: [numClusters][numCoords] */
    float  **dimClusters;
    float  **newClusters;    /* [numCoords][numClusters] */

    float *deviceObjects;
    float *deviceClusters;
    int *deviceMembership;
    int *deviceIntermediates;

    malloc2D(dimObjects, numCoords, numObjs, float);
    for (i = 0; i < numCoords; i++) {
        for (j = 0; j < numObjs; j++) {
            dimObjects[i][j] = objects[j][i];
        }
    }

    malloc2D(dimClusters, numCoords, numClusters, float);
    for (i = 0; i < numCoords; i++) {
        for (j = 0; j < numClusters; j++) {
            dimClusters[i][j] = dimObjects[i][j];
        }
    }

    for (i=0; i<numObjs; i++) membership[i] = -1;

    newClusterSize = (int*) calloc(numClusters, sizeof(int));
    assert(newClusterSize != NULL);

    malloc2D(newClusters, numCoords, numClusters, float);
    memset(newClusters[0], 0, numCoords * numClusters * sizeof(float));

    const unsigned int numThreadsPerClusterBlock = 128;
    const unsigned int numClusterBlocks =
        (numObjs + numThreadsPerClusterBlock - 1) / numThreadsPerClusterBlock;
#if BLOCK_SHARED_MEM_OPTIMIZATION
    const unsigned int clusterBlockSharedDataSize =
        numThreadsPerClusterBlock * sizeof(unsigned char) +
        numClusters * numCoords * sizeof(float);

    cudaDeviceProp deviceProp;
    int deviceNum;
    cudaGetDevice(&deviceNum);
    cudaGetDeviceProperties(&deviceProp, deviceNum);

    if (clusterBlockSharedDataSize > deviceProp.sharedMemPerBlock) {
        err("Insuficiente memoria compartida para los bloques\n");
    }
#else
    const unsigned int clusterBlockSharedDataSize =
        numThreadsPerClusterBlock * sizeof(unsigned char);
#endif

    const unsigned int numReductionThreads =
        nextPowerOfTwo(numClusterBlocks);
    const unsigned int reductionBlockSharedDataSize =
        numReductionThreads * sizeof(unsigned int);

    checkCuda(cudaMalloc(&deviceObjects, numObjs*numCoords*sizeof(float)));
    checkCuda(cudaMalloc(&deviceClusters, numClusters*numCoords*sizeof(float)));
    checkCuda(cudaMalloc(&deviceMembership, numObjs*sizeof(int)));
    checkCuda(cudaMalloc(&deviceIntermediates, numReductionThreads*sizeof(unsigned int)));

  // copiamos los datos de los puntos y los centros aleatorios en el dispositivo

    checkCuda(cudaMemcpy(deviceObjects, dimObjects[0], numObjs*numCoords*sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(deviceMembership, membership, numObjs*sizeof(int), cudaMemcpyHostToDevice));

    do {
        checkCuda(cudaMemcpy(deviceClusters, dimClusters[0],
                  numClusters*numCoords*sizeof(float), cudaMemcpyHostToDevice));

  // se inicializa el Kernel con: Num de bloques por cluster, num de hilos por bloque de cluster,
  // tamaño de datos de los bloqes de cluster compartidos

        find_nearest_cluster
            <<< numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize >>>
            (numCoords, numObjs, numClusters,
             deviceObjects, deviceClusters, deviceMembership, deviceIntermediates);

        cudaDeviceSynchronize(); checkLastCudaError();

        compute_delta <<< 1, numReductionThreads, reductionBlockSharedDataSize >>>
            (deviceIntermediates, numClusterBlocks, numReductionThreads);
// se espera a que el dispositivo termine de procesar los datos
        cudaDeviceSynchronize(); checkLastCudaError();

// se copian los resultados en el host

        int d;
        checkCuda(cudaMemcpy(&d, deviceIntermediates, sizeof(int), cudaMemcpyDeviceToHost));
        delta = (float)d;

        checkCuda(cudaMemcpy(membership, deviceMembership, numObjs*sizeof(int), cudaMemcpyDeviceToHost));

        for (i=0; i<numObjs; i++) {
            /* encontrar el indice del cetro del cluster */
            index = membership[i];

            /* actualizar los nuevos centros */
            newClusterSize[index]++;
            for (j=0; j<numCoords; j++)
                newClusters[j][index] += objects[i][j];
        }

        //  TODO: Cambiar el orden de los ppuntos
        //  TODO: [numClusters][numCoords]
        /* suma promedio y reemplazar los antiguos clusters*/
        for (i=0; i<numClusters; i++) {
            for (j=0; j<numCoords; j++) {
                if (newClusterSize[i] > 0)
                    dimClusters[j][i] = newClusters[j][i] / newClusterSize[i];
                newClusters[j][i] = 0.0;
            }
            newClusterSize[i] = 0;
        }

        delta /= numObjs;
    } while (delta > threshold && loop++ < 500);

    *loop_iterations = loop + 1;

    // asignar un espacio 2d para las coordenadas de los clusters

    malloc2D(clusters, numClusters, numCoords, float);
    for (i = 0; i < numClusters; i++) {
        for (j = 0; j < numCoords; j++) {
            clusters[i][j] = dimClusters[j][i];
        }
    }

    checkCuda(cudaFree(deviceObjects));
    checkCuda(cudaFree(deviceClusters));
    checkCuda(cudaFree(deviceMembership));
    checkCuda(cudaFree(deviceIntermediates));

    free(dimObjects[0]);
    free(dimObjects);
    free(dimClusters[0]);
    free(dimClusters);
    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);

    return clusters;
}

