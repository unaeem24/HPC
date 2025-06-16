#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>

#define MAX_ITER 1000
#define TOLERANCE 1e-6

// Serial implementation
void serial_laplace(int N, double **grid, double **new_grid) {
    double diff, delta;
    int iter = 0;
    
    do {
        diff = 0.0;
        
        for (int i = 1; i < N-1; i++) {
            for (int j = 1; j < N-1; j++) {
                new_grid[i][j] = 0.25 * (grid[i-1][j] + grid[i+1][j] + 
                                         grid[i][j-1] + grid[i][j+1]);
                delta = fabs(new_grid[i][j] - grid[i][j]);
                if (delta > diff) diff = delta;
            }
        }
        
        // Swap grids
        double **temp = grid;
        grid = new_grid;
        new_grid = temp;
        
        iter++;
    } while (diff > TOLERANCE && iter < MAX_ITER);
}

// OpenMP implementation
void openmp_laplace(int N, double **grid, double **new_grid) {
    double diff, global_diff;
    int iter = 0;
    
    do {
        global_diff = 0.0;
        
        #pragma omp parallel private(diff) shared(global_diff)
        {
            diff = 0.0;
            #pragma omp for schedule(static)
            for (int i = 1; i < N-1; i++) {
                for (int j = 1; j < N-1; j++) {
                    new_grid[i][j] = 0.25 * (grid[i-1][j] + grid[i+1][j] + 
                                             grid[i][j-1] + grid[i][j+1]);
                    double delta = fabs(new_grid[i][j] - grid[i][j]);
                    if (delta > diff) diff = delta;
                }
            }
            
            #pragma omp critical
            {
                if (diff > global_diff) global_diff = diff;
            }
        }
        
        // Swap grids
        double **temp = grid;
        grid = new_grid;
        new_grid = temp;
        
        iter++;
    } while (global_diff > TOLERANCE && iter < MAX_ITER);
}

// MPI implementation
void mpi_laplace(int N, double **grid, double **new_grid, int rank, int size) {
    double diff, global_diff;
    int iter = 0;
    
    // Calculate rows per process
    int rows_per_proc = (N-2) / size;
    int remainder = (N-2) % size;
    
    int start_row = 1 + rank * rows_per_proc + (rank < remainder ? rank : remainder);
    int end_row = start_row + rows_per_proc + (rank < remainder ? 1 : 0);
    
    // Allocate buffers for halo exchange
    double *top_send = (double *)malloc(N * sizeof(double));
    double *top_recv = (double *)malloc(N * sizeof(double));
    double *bottom_send = (double *)malloc(N * sizeof(double));
    double *bottom_recv = (double *)malloc(N * sizeof(double));
    
    MPI_Request requests[4];
    
    do {
        global_diff = 0.0;
        diff = 0.0;
        
        // Prepare halo data for sending
        if (rank > 0) {
            for (int j = 0; j < N; j++) {
                top_send[j] = grid[start_row][j];
            }
            MPI_Isend(top_send, N, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &requests[0]);
            MPI_Irecv(top_recv, N, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, &requests[1]);
        }
        
        if (rank < size-1) {
            for (int j = 0; j < N; j++) {
                bottom_send[j] = grid[end_row-1][j];
            }
            MPI_Isend(bottom_send, N, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &requests[2]);
            MPI_Irecv(bottom_recv, N, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &requests[3]);
        }
        
        // Compute interior points that don't need halo data
        for (int i = start_row+1; i < end_row-1; i++) {
            for (int j = 1; j < N-1; j++) {
                new_grid[i][j] = 0.25 * (grid[i-1][j] + grid[i+1][j] + 
                                         grid[i][j-1] + grid[i][j+1]);
                double delta = fabs(new_grid[i][j] - grid[i][j]);
                if (delta > diff) diff = delta;
            }
        }
        
        // Complete halo exchange
        if (rank > 0) {
            MPI_Wait(&requests[0], MPI_STATUS_IGNORE);
            MPI_Wait(&requests[1], MPI_STATUS_IGNORE);
            
            // Update top boundary with received halo
            int i = start_row;
            for (int j = 1; j < N-1; j++) {
                new_grid[i][j] = 0.25 * (top_recv[j] + grid[i+1][j] + 
                                     grid[i][j-1] + grid[i][j+1]);
                double delta = fabs(new_grid[i][j] - grid[i][j]);
                if (delta > diff) diff = delta;
            }
        }
        
        if (rank < size-1) {
            MPI_Wait(&requests[2], MPI_STATUS_IGNORE);
            MPI_Wait(&requests[3], MPI_STATUS_IGNORE);
            
            // Update bottom boundary with received halo
            int i = end_row-1;
            for (int j = 1; j < N-1; j++) {
                new_grid[i][j] = 0.25 * (grid[i-1][j] + bottom_recv[j] + 
                                     grid[i][j-1] + grid[i][j+1]);
                double delta = fabs(new_grid[i][j] - grid[i][j]);
                if (delta > diff) diff = delta;
            }
        }
        
        // Reduce diff across all processes
        MPI_Allreduce(&diff, &global_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        
        // Swap grids
        double **temp = grid;
        grid = new_grid;
        new_grid = temp;
        
        iter++;
    } while (global_diff > TOLERANCE && iter < MAX_ITER);
    
    free(top_send);
    free(top_recv);
    free(bottom_send);
    free(bottom_recv);
}

// Allocate 2D array
double **allocate_grid(int N) {
    double **grid = (double **)malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++) {
        grid[i] = (double *)malloc(N * sizeof(double));
    }
    return grid;
}

// Initialize grid with boundary conditions
void initialize_grid(int N, double **grid) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            grid[i][j] = 0.0;
        }
    }
    
    // Set boundary conditions
    for (int i = 0; i < N; i++) {
        grid[i][0] = 100.0;   // Left boundary
        grid[i][N-1] = 100.0; // Right boundary
        grid[0][i] = 0.0;     // Top boundary
        grid[N-1][i] = 0.0;   // Bottom boundary
    }
}

// Free allocated memory
void free_grid(int N, double **grid) {
    for (int i = 0; i < N; i++) {
        free(grid[i]);
    }
    free(grid);
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        printf("Grid(NxN)\tSerial Time(s)\tOpenMP Time(ms)\tMPI Time(s)\n");
        printf("----------------------------------------------------\n");
    }
    
    for (int N = 100; N <= 1000; N += 100) {
        double **grid, **new_grid, **grid_copy;
        double serial_time, openmp_time, mpi_time;
        
        if (rank == 0) {
            // Allocate and initialize grids
            grid = allocate_grid(N);
            new_grid = allocate_grid(N);
            grid_copy = allocate_grid(N);
            initialize_grid(N, grid);
            initialize_grid(N, new_grid);
            initialize_grid(N, grid_copy);
            
            // Serial execution
            clock_t start = clock();
            serial_laplace(N, grid, new_grid);
            serial_time = (double)(clock() - start) / CLOCKS_PER_SEC;
            
            // Reset grids for OpenMP
            initialize_grid(N, grid);
            initialize_grid(N, new_grid);
            
            // OpenMP execution
            start = clock();
            openmp_laplace(N, grid, new_grid);
            openmp_time = (double)(clock() - start) / CLOCKS_PER_SEC;
            
            // Reset grids for MPI
            initialize_grid(N, grid);
            initialize_grid(N, new_grid);
        }
        
        // MPI execution - all processes participate
        MPI_Barrier(MPI_COMM_WORLD);
        double mpi_start = MPI_Wtime();
        
        if (rank == 0) {
            // Broadcast initial grid to all processes
            for (int i = 0; i < N; i++) {
                MPI_Bcast(grid[i], N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Bcast(new_grid[i], N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            }
        } else {
            // Other processes allocate memory
            grid = allocate_grid(N);
            new_grid = allocate_grid(N);
            
            // Receive broadcasted grid
            for (int i = 0; i < N; i++) {
                MPI_Bcast(grid[i], N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Bcast(new_grid[i], N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            }
        }
        
        mpi_laplace(N, grid, new_grid, rank, size);
        
        // Gather results to rank 0 if needed
        // (omitted for brevity, but would use MPI_Gatherv)
        
        MPI_Barrier(MPI_COMM_WORLD);
        double mpi_end = MPI_Wtime();
        
        if (rank == 0) {
            mpi_time = mpi_end - mpi_start;
            printf("%dx%d\t\t%.6f\t%.6f\t%.6f\n", N, N, serial_time, openmp_time, mpi_time);
            
            // Free memory
            free_grid(N, grid);
            free_grid(N, new_grid);
            free_grid(N, grid_copy);
        } else {
            free_grid(N, grid);
            free_grid(N, new_grid);
        }
    }
    
    MPI_Finalize();
    return 0;
}