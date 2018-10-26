#include <mpi.h>
#include <iostream>


int main(int argc, char** argv) {
 
  int rank, numtasks, receivedRank, tag=1;

  MPI_Status Stat; // required variable for receive routines
  MPI_Init(nullptr, nullptr); 
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  double elapsed_time(0);

  int prevRank = rank - 1;
  int nextRank = rank + 1;
  if(rank == 0) prevRank = numtasks - 1;
  if (rank == numtasks - 1) nextRank = 0;

  double t1 = MPI_Wtime(); 
  for(int i = 0; i < 1000; i++){
  MPI_Send(&rank, 1, MPI_INT, prevRank, tag, MPI_COMM_WORLD);
  MPI_Send(&rank, 1, MPI_INT, nextRank, tag, MPI_COMM_WORLD);
  MPI_Recv(&receivedRank, 1, MPI_INT, prevRank, tag, MPI_COMM_WORLD, &Stat);
  //std::cout << "Rank " << rank << " received message from rank " << receivedRank << '\n';
  MPI_Recv(&receivedRank, 1, MPI_INT, nextRank, tag, MPI_COMM_WORLD, &Stat);  
  //std::cout << "Rank " << rank << " received message from rank " << receivedRank << '\n';
  }

  elapsed_time = MPI_Wtime() - t1;
  MPI_Finalize();

  if(rank == 0)
    std::cout << "Elapsed time: " << elapsed_time << std::endl;

  return 0;

}
