// Threaded two-dimensional Discrete FFT transform
// Neha Kadam
// ECE6122 Project 2

#include <iostream>
#include <string>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include "Complex.h"
#include "InputImage.h"

// You will likely need global variables indicating how
// many threads there are, and a Complex* that points to the
// 2d image being transformed.

int ih, iw;
int N = 1024;
pthread_mutex_t exitMutex, printMutex, threadCountMutex;
int numThreads = 16;
int running = 0;
Complex *h = new Complex[N * N];
Complex *weights = new Complex[N/2];
pthread_cond_t exitCond;
bool flag; /* 1 for fwd and 0 for inverse*/


void reorderArray(Complex *d);


/* For myBarrier */
int P = numThreads + 1;
int count;
pthread_mutex_t countMutex;
bool *localSense;
bool globalSense;
int FetchAndDecrementCount();

using namespace std;

/*Function Declarations */
void MyBarrier_Init();
void MyBarrier(int myId);
void calcWeights();
void Transpose(Complex* data);
void Transform1D(Complex* h, int N);
void reorderArray(Complex *d);
void* Transform2DTHread(void* v);
void Transform2D(const char* inputFN);

// Function to reverse bits in an unsigned integer
// This assumes there is a global variable N that is the
// number of points in the 1D transform.
unsigned ReverseBits(unsigned v)
{ //  Provided to students
  unsigned n = N; // Size of array (which is even 2 power k value)
  unsigned r = 0; // Return value
   
  for (--n; n > 0; n >>= 1)
    {
      r <<= 1;        // Shift return value
      r |= (v & 0x1); // Merge in next bit
      v >>= 1;        // Shift reversal value
    }
  return r;
}

// GRAD Students implement the following 2 functions.
// Undergrads can use the built-in barriers in pthreads.

// Call MyBarrier_Init once in main

int FetchAndDecrementCount()
{
	pthread_mutex_lock(&countMutex);
	int myCount = count;
	count--;
	pthread_mutex_unlock(&countMutex);
	return myCount;
}

void MyBarrier_Init()// you will likely need some parameters)
{
	count = P;
	
	pthread_mutex_init(&countMutex, 0);

	localSense = new bool[P];
	for(int i = 0; i < P; i++)
	{
		localSense[i] = true;
	}
	globalSense = true;

}

// Each thread calls MyBarrier after completing the row-wise DFT
void MyBarrier(int myId) // Again likely need parameters
{
	
	localSense[myId] = !localSense[myId];
	
	if(FetchAndDecrementCount() == 1)
	{
		count = P;
		globalSense = localSense[myId];
	}
	else
	{
		while(globalSense != localSense[myId]) {}
	}
}

void calcWeights()
{
	
	if(flag == 1) 	/* fwd */	
	{
		for(int i = 0; i < N/2; i++)
        	{
			double real, imag;
			
			real = cos(2 * M_PI * i / N);
			imag = -sin(2 * M_PI * i / N);

			weights[i].real = real;
			weights[i].imag = imag;
        	}
	}

	if(flag == 0) 	/* inverse */	
	{
		for(int i = 0; i < N/2; i++)
        	{
			             
			double real, imag;
			
			real = cos(2 * M_PI * i / N);
			imag = sin(2 * M_PI * i / N);

			weights[i].real = real;
			weights[i].imag = imag;
        	       
        	}
	}
}
  
void Transpose(Complex* data)
{
        for(int row = 0; row < N; ++row)
        {
                for(int col = 0; col < N; ++col)
                {
                        if(row < col)
                        {
                                Complex temp = data[N * row + col];
                                data[N * row + col] = data[N * col + row];
                                data[N * col + row] = temp;
                        }
                }
        }
}

               
void Transform1D(Complex* h, int N)
{
  // Implement the efficient Danielson-Lanczos DFT here.
  // "h" is an input/output parameter
  // "N" is the size of the array (assume even power of 2)

	int k, wMatrixInd;

	for(int numOfPoints = 1; numOfPoints < N; numOfPoints *= 2)
	{
	        for(int i = 0; i < numOfPoints; ++i)
	        {
	                wMatrixInd = i * N / (numOfPoints * 2);
	
		        for(int j = i; j < N; j += (numOfPoints*2))
		        {
		                k = j + numOfPoints;

		                Complex temp = weights[wMatrixInd] * h[k];
		                h[k] = h[j] - temp;
		                h[j] = h[j] + temp;
		        }

		}
	}


	if (flag == 0) // for inverse 
	{
		for(int i = 0; i < N; i++)
		{
			h[i].real = h[i].real/N;
			h[i].imag = h[i].imag/N;			
			if(fabs(h[i].real) < 1e-12) h[i].real=0;
			if(fabs(h[i].imag) < 1e-12) h[i].imag=0;
		}

			
	}


}

void reorderArray(Complex *d)
{
        int j;
        for(int i = 0; i < N; ++i)
        {
                j = ReverseBits(i);
                if (j > i)
                {
                        Complex tmp = d[i];
                        d[i] = d[j];
                        d[j] = tmp;
                }
        }
}

void* Transform2DThread(void* v)
{ // This is the thread startign point.  "v" is the thread number
  // Calculate 1d DFT for assigned rows
  // wait for all to complete
  // Calculate 1d DFT for assigned columns
  // Decrement active count and signal main if all complete
	
	uint64_t thread_num = (uint64_t) v;
	int rank = (int) thread_num;
	int rowsPerThread = ih / numThreads;
	int startRow = rowsPerThread * rank;
	

	for(int i = startRow; i < (startRow + rowsPerThread); i++)
	{
		reorderArray(&h[i * iw]);
		Transform1D(&h[i * iw], N);
	}

	MyBarrier(rank);

	pthread_mutex_lock(&threadCountMutex);
	
	if(--running == 0)
	{
		pthread_mutex_lock(&exitMutex);
		pthread_cond_signal(&exitCond);
		pthread_mutex_unlock(&exitMutex);
	}
	pthread_mutex_unlock(&threadCountMutex);
	
	return (void*)0;
}

void CreateThreads()
{

	for(int i = 0; i < numThreads; i++)
	{
		pthread_mutex_lock(&threadCountMutex);
		running++;
		pthread_mutex_unlock(&threadCountMutex);

		pthread_t pt;
		pthread_create(&pt, 0, Transform2DThread, (void*)i);
	}
}

void Transform2D(const char* inputFN) 
{ 
	// Do the 2D transform here.
	InputImage image(inputFN);  // Create the helper object for reading the image
  
	// Create the global pointer to the image array data
	h = image.GetImageData();
	ih = image.GetHeight();
	iw = image.GetWidth();


	flag = 1;	
	calcWeights();

	// Create 16 threads
	CreateThreads();
	
	MyBarrier(numThreads);

	pthread_cond_wait(&exitCond,&exitMutex);
//	image.SaveImageData("myafter1d.txt", h, N, N);

	
	Transpose(h);


	running = 0;
	
	CreateThreads();

	MyBarrier(numThreads);

	
	Transpose(h);
	

	pthread_cond_wait(&exitCond,&exitMutex);

	image.SaveImageData("Tower-DFT2D.txt", h, N, N);

  
	cout << "Fwd transform done! "<< endl;


	/* Now for Inverse*/
	flag = 0;
	calcWeights();


	running = 0;


	CreateThreads();
	MyBarrier(numThreads);
	pthread_cond_wait(&exitCond,&exitMutex);
//	image.SaveImageData("myafterInverse.txt", h, N, N);


	Transpose(h);


	running = 0;


	CreateThreads();
	MyBarrier(numThreads);
	
	Transpose(h);
	

	pthread_cond_wait(&exitCond,&exitMutex);
	image.SaveImageData("TowerInverse.txt", h, N, N);

	delete [] weights;
	delete [] h;

}

int main(int argc, char** argv)
{
  string fn("Tower.txt"); // default file name
  if (argc > 1) fn = string(argv[1]);  // if name specified on cmd line
  // MPI initialization here


  	MyBarrier_Init();

	//Initialize mutexes
	pthread_mutex_init(&exitMutex, 0);
	pthread_mutex_init(&printMutex, 0);
	pthread_mutex_init(&threadCountMutex, 0);
	pthread_cond_init(&exitCond, 0);

	//lock exit mutex 
	pthread_mutex_lock(&exitMutex);

	Transform2D(fn.c_str()); // Perform the transform.
}  
  

  

