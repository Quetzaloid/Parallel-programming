// Parallel version of Gaussian Elimination
// Solve the equation: matrix * X = R
#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>
#include <assert.h>

using namespace std;


//#define TEST 
#define SWAP(a,b)   {double tmp; tmp = a; a = b; b = tmp;}

int         proc_id, nprocs;
int         NumSize = 0;
int         NumAlloc = 0;
int         elimIdx = 0;
int         tasAlloc[32];
double    **matrix, *X, *R;
double     *X__;
double     *pivotRow;
double    **delivMat;

/* For all the rows, get the pivot and eliminate all rows and columns for that particular pivot row. */
void    mas_fn(); 
void    sla_fn();

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

/*Attach/detach results. */
void attachR(int nsize);
void detachR(int nsize);


int main(int argc, char *argv[])
{
    struct timeval start, finish;
    double error;
    int namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
    MPI_Get_processor_name(processor_name, &namelen);

    printf("MPI environment established! Processor %d of %d on %s\n", proc_id, nprocs, processor_name);
    fflush(stdout);

    if (argc < 2) 
       {
         fprintf(stderr, "usage: %s <matrixfile>\n", argv[0]);
         exit(1);
       }

    if (proc_id==0)
       {
         NumSize = initMatrix(argv[1]);
         NumAlloc = NumSize/nprocs + ((NumSize%nprocs>0)? 1:0);
         initRHS(NumSize);
         initResult(NumSize);
         attachR(NumSize);
       }

    MPI_Bcast(&NumSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NumAlloc, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (proc_id!=0)
       {
         double *tmp;
         delivMat = (double**)malloc(NumAlloc*sizeof(double*)); 
         assert(delivMat != NULL);
         tmp = (double*)malloc(NumSize*(NumSize+1)*sizeof(double));
         assert(tmp != NULL);

         for (int i = 0; i < NumAlloc; i++) {
             delivMat[i] = tmp;
             tmp = tmp + NumSize + 1;
         }

         /////////////////////////
         if ((pivotRow = (double *) malloc(sizeof(double)*(NumSize+1))) == 0) 
            {
              fprintf(stderr,"out of shared memory");
              fflush(stderr);
              MPI_Finalize(); 
              exit(-1);
            }
       }

    MPI_Barrier(MPI_COMM_WORLD);

    gettimeofday(&start, 0);
    //////////////////////////////////////////////////////
    ///////////////GAUSSIAN ELIMINATION///////////////////
    //////////////////////////////////////////////////////
    if (proc_id==0)
       {
         mas_fn();
       }
    else
       {
         sla_fn();
       }
    MPI_Barrier(MPI_COMM_WORLD);
    ///////////////////////////////////////////////////////
    gettimeofday(&finish, 0);
    MPI_Barrier(MPI_COMM_WORLD);

    if (proc_id==0)
       {
         detachR(NumSize);
         solveGauss(NumSize);
         
/*         for(int i = 0; i < NumSize; i++)
            {
              for(int j = 0; j < NumSize+1; j++)
                 {
                   cout<<matrix[i][j]<<" ";
                 }
              cout<<endl;
            }
*/ 
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
       }
    MPI_Barrier(MPI_COMM_WORLD);
    return 0;
}

void  mas_fn()
{
    MPI_Status  status;
    MPI_Request request;
    
    for(int i=0; i<NumSize; i++) 
       {         
         getPivot(NumSize, i);
         double pivotval = matrix[i][i];
         for(int j=i; j<=NumSize; j++)
            {
              matrix[i][j] /= pivotval;
            }
         
         int Quotient = (NumSize-i-1)/nprocs;
         int Remaindr = (NumSize-i-1)%nprocs;

         for(int j=0; j<nprocs; j++)
            {
              tasAlloc[j] = Quotient;
            }
         for(int j=0; j<Remaindr; j++)
            {
              tasAlloc[j]++;
            }

         elimIdx = i;
         MPI_Bcast(tasAlloc, 32, MPI_INT, 0, MPI_COMM_WORLD);
         MPI_Bcast(&elimIdx, 1, MPI_INT, 0, MPI_COMM_WORLD);

         for(int j=1; j<nprocs; j++)
            {
              MPI_Isend(matrix[i], NumSize+1, MPI_DOUBLE, j, 0, MPI_COMM_WORLD, &request);
            }
 
         int curIdx = i+1+tasAlloc[0];
         for(int j=1; j<nprocs; j++)
            {
              MPI_Isend(matrix[curIdx], tasAlloc[j]*(NumSize+1), MPI_DOUBLE, j, 0, MPI_COMM_WORLD, &request);
              curIdx += tasAlloc[j];
            }

         for(int j=i+1; j<i+1+tasAlloc[0]; j++)
            {
              pivotval = matrix[j][i];
              for(int k=i; k<NumSize+1; k++)
                 {
                   matrix[j][k] -= pivotval*matrix[i][k];
                 }
            }

         curIdx = i+1+tasAlloc[0];
         for(int j=1; j<nprocs; j++)
            {
              MPI_Recv(matrix[curIdx], tasAlloc[j]*(NumSize+1), MPI_DOUBLE, j, 3, MPI_COMM_WORLD, &status);
              curIdx += tasAlloc[j];
            }
       }
}

void  sla_fn()
{
    MPI_Status  status;
    MPI_Request request;

    for(int n=0; n<NumSize; n++)
       { 
         MPI_Bcast(tasAlloc, 32, MPI_INT, 0, MPI_COMM_WORLD);
         MPI_Bcast(&elimIdx, 1, MPI_INT, 0, MPI_COMM_WORLD);
         MPI_Recv(pivotRow, NumSize+1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
         MPI_Recv(delivMat[0], tasAlloc[proc_id]*(NumSize+1), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
         for(int i=0; i<tasAlloc[proc_id]; i++)
            {
              double pivotval = delivMat[i][elimIdx];
              for(int k=elimIdx; k<NumSize+1; k++)
                 {
                   delivMat[i][k] -= pivotval*pivotRow[k];
                 }
            }
         MPI_Ssend(delivMat[0], tasAlloc[proc_id]*(NumSize+1), MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
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
    tmp = (double*)malloc(nsize*(nsize+1)*sizeof(double));
    assert(tmp != NULL);    
    for (i = 0; i < nsize; i++) {
        matrix[i] = tmp;
        tmp = tmp + nsize + 1;
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
        for (i = currow; i <= nsize; i++) {
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

void attachR(int nsize)
{
    for(int i=0; i<nsize; i++)
       {
         matrix[i][nsize] = R[i];
       }
}

void detachR(int nsize)
{
    for(int i=0; i<nsize; i++)
       {
         R[i] = matrix[i][nsize];
       }
}
