// Parallel version of Gaussian Elimination
// Solve the equation: matrix * X = R
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>
#include <assert.h>

using namespace std;

#define SWAP(a,b)       {double tmp; tmp = a; a = b; b = tmp;}

pthread_t  *ThParam;
pthread_t  *ThHandle;

int         NumThreads = 1;
int         NumSize    = 0;
double    **matrix, *X, *R;
double     *X__;

pthread_mutex_t mut = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t  cond = PTHREAD_COND_INITIALIZER;

typedef struct MAT_CT
{ 
    int count;
    pthread_mutex_t lock;
}   MAT_CT;

MAT_CT mat_count;

void barrier (int curLocus);

/* For all the rows, get the pivot and eliminate all rows and columns for that particular pivot row. */
void  *thr_fn(void *thIdx);

/* Initialize the matirx. */
int  initMatrix(const char *fname);

/* Initialize the right-hand-side following the pre-set solution. */
void initRHS(int nsize);

/* Initialize the results. */
void initResult(int nsize);

/* Get the pivot - the element on column with largest absolute value. */
void getPivot(int nsize, int currow);

/* Solve the equation. */
void solveGauss(int nsize);


int main(int argc, char *argv[])
{
    struct timeval start, finish;
    double error;
    
    if (argc < 2) 
       {
         fprintf(stderr, "usage: %s <matrixfile>\n", argv[0]);
         exit(1);
       }
    else
       {
         NumThreads = atoi(argv[2]);
       }

    NumSize = initMatrix(argv[1]);
    initRHS(NumSize);
    initResult(NumSize);

    ThParam  = (pthread_t *)malloc(sizeof(pthread_t)*NumThreads);
    ThHandle = (pthread_t *)malloc(sizeof(pthread_t)*NumThreads);
    if (!ThParam || !ThHandle)  
       {
         cout<<"!!!!ERROR: MEMORY ALLOCATING FAILS!!!"<<endl;
         exit(1);
       }

    int ThCreate;
    pthread_attr_t attr;
    pthread_attr_init (&attr);
    pthread_attr_setscope (&attr, PTHREAD_SCOPE_SYSTEM);

    pthread_mutex_init(&(mat_count.lock),NULL);

    gettimeofday(&start, 0);
    //////////////////////////////////////////////////////
    ///////////////GAUSSIAN ELIMINATION///////////////////
    //////////////////////////////////////////////////////
    for(int i=0; i<NumThreads; i++)
       {
         ThParam[i] = i;
         ThCreate   = pthread_create(&ThHandle[i], &attr, thr_fn, (void *)&ThParam[i]);
         if (ThCreate != 0)
            {
              cout<<"!!!!!THREAD "<<i<<" CREATION FAILED ... Exiting abruptly"<<endl<<endl;
              exit(1);
            }
       }
    for(int i=0; i<NumThreads; i++)
       {
         pthread_join(ThHandle[i], NULL);
       }
    ///////////////////////////////////////////////////////
    gettimeofday(&finish, 0);

    solveGauss(NumSize);
    
    fprintf(stdout, "Time:  %f seconds\n", (finish.tv_sec - start.tv_sec) + (finish.tv_usec - start.tv_usec)*0.000001);
    
    error = 0.0;
    for(int i = 0; i < NumSize; i++) 
       {
         double error__ = (X__[i]==0.0) ? 1.0 : fabs((X[i]-X__[i])/X__[i]);
         if (error < error__) 
            {
              error = error__;
            }
       }
    fprintf(stdout, "Error: %e\n", error);

    return 0;
}

void barrier(int curLocus)
{
    static int arrived = 0;

    pthread_mutex_lock (&mut);  //lock

    arrived++;
    if (arrived < NumThreads)
        pthread_cond_wait (&cond, &mut);
    else {
        arrived = 0;    // reset the barrier before broadcast is important
        mat_count.count = curLocus;
        pthread_cond_broadcast (&cond);
    }

    pthread_mutex_unlock (&mut);  //unlock
}

void  *thr_fn(void *thIdx)
{
    double pivotval;

    int tIndex = *((int *) thIdx);
    
    for(int i=0; i<NumSize; i++) 
       {         
         if (i%NumThreads == tIndex)
            {
              getPivot(NumSize, i);
              pivotval = matrix[i][i];
              for(int j=i; j<NumSize; j++)
                 {
                   matrix[i][j] /= pivotval;
                 }
              R[i] /= pivotval;
            }
         barrier(i);
         
         int j = i;
         while(j<NumSize)
              {
                pthread_mutex_lock(&(mat_count.lock));              
                mat_count.count++; 
                j = mat_count.count;
                pthread_mutex_unlock(&(mat_count.lock));
                if (j<NumSize)
                   {
                     pivotval = matrix[j][i];
                     for(int k=i; k<NumSize; k++)
                        {
                          matrix[j][k] -= pivotval*matrix[i][k];
                        }
                     R[j] -= pivotval * R[i];
                   }
              }
         barrier(i);
       }
}

int initMatrix(const char *fname)
{
    FILE *file;
    int l1, l2, l3;
    double d;
    int nsize;
    int i, j;
    double *tmp;
    char buffer[1024];

    if ((file = fopen(fname, "r")) == NULL) {
    fprintf(stderr, "The matrix file open error\n");
        exit(-1);
    }
    
    /* Parse the first line to get the matrix size. */
    fgets(buffer, 1024, file);
    sscanf(buffer, "%d %d %d", &l1, &l2, &l3);
    nsize = l1;
#ifdef DEBUG
    fprintf(stdout, "matrix size is %d\n", nsize);
#endif

    /* Initialize the space and set all elements to zero. */
    matrix = (double**)malloc(nsize*sizeof(double*));
    assert(matrix != NULL);
    tmp = (double*)malloc(nsize*nsize*sizeof(double));
    assert(tmp != NULL);    
    for (i = 0; i < nsize; i++) {
        matrix[i] = tmp;
        tmp = tmp + nsize;
    }
    for (i = 0; i < nsize; i++) {
        for (j = 0; j < nsize; j++) {
            matrix[i][j] = 0.0;
        }
    }

    /* Parse the rest of the input file to fill the matrix. */
    for (;;) {
    fgets(buffer, 1024, file);
    sscanf(buffer, "%d %d %lf", &l1, &l2, &d);
    if (l1 == 0) break;

    matrix[l1-1][l2-1] = d;
#ifdef DEBUG
    fprintf(stdout, "row %d column %d of matrix is %e\n", l1-1, l2-1, matrix[l1-1][l2-1]);
#endif
    }

    fclose(file);
    return nsize;
}

void initRHS(int nsize)
{
    int i, j;

    X__ = (double*)malloc(nsize * sizeof(double));
    assert(X__ != NULL);
    for (i = 0; i < nsize; i++) {
    X__[i] = i+1;
    }

    R = (double*)malloc(nsize * sizeof(double));
    assert(R != NULL);
    for (i = 0; i < nsize; i++) {
    R[i] = 0.0;
    for (j = 0; j < nsize; j++) {
        R[i] += matrix[i][j] * X__[j];
    }
    }
}

void initResult(int nsize)
{
    int i;

    X = (double*)malloc(nsize * sizeof(double));
    assert(X != NULL);
    for (i = 0; i < nsize; i++) {
    X[i] = 0.0;
    }
}

void getPivot(int nsize, int currow)
{
    int i, pivotrow;

    pivotrow = currow;
    for (i = currow+1; i < nsize; i++) {
    if (fabs(matrix[i][currow]) > fabs(matrix[pivotrow][currow])) {
        pivotrow = i;
    }
    }

    if (fabs(matrix[pivotrow][currow]) == 0.0) {
        fprintf(stderr, "The matrix is singular\n");
        exit(-1);
    }
    
    if (pivotrow != currow) {
#ifdef DEBUG
    fprintf(stdout, "pivot row at step %5d is %5d\n", currow, pivotrow);
#endif
        for (i = currow; i < nsize; i++) {
            SWAP(matrix[pivotrow][i],matrix[currow][i]);
        }
        SWAP(R[pivotrow],R[currow]);
    }
}

void solveGauss(int nsize)
{
    int i, j;

    X[nsize-1] = R[nsize-1];
    for (i = nsize - 2; i >= 0; i --) {
        X[i] = R[i];
        for (j = nsize - 1; j > i; j--) {
            X[i] -= matrix[i][j] * X[j];
        }
    }

#ifdef DEBUG
    fprintf(stdout, "X = [");
    for (i = 0; i < nsize; i++) {
        fprintf(stdout, "%.6f ", X[i]);
    }
    fprintf(stdout, "];\n");
#endif
}
