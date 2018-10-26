#include <mpi.h>
#include <iostream>


int main(int argc, char** argv) {
 
  int rank, numtasks, source, dest, count, tag=1;
  int msg_size = 5;
  char inmsg[msg_size];
  char outmsg[msg_size] = "Hello";

  MPI_Status Stat; // required variable for receive routines
  MPI_Init(nullptr, nullptr); 
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  double elapsed_time(0);

  double t1 = MPI_Wtime(); 
  for(int i = 0; i < 10000; i++){
    // task 0 sends a message to task 1 and waits to get a return message
    if (rank == 0){
      dest = 1;
      MPI_Send(&outmsg, msg_size, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
      MPI_Recv(&inmsg, msg_size, MPI_CHAR, dest, tag, MPI_COMM_WORLD, &Stat);  
      //std::cout << "Message exchanged on rank between rank " << rank << " and rank " << dest << " : " << inmsg << '\n';
    }
    
    // task 1 waits to receive a message from task 0 and return a message
    else if(rank == 1){
      source = 0;
      MPI_Recv(&inmsg, msg_size, MPI_CHAR, source, tag, MPI_COMM_WORLD, &Stat);
      MPI_Send(&outmsg, msg_size, MPI_CHAR, source, tag, MPI_COMM_WORLD);
    }
    MPI_Get_count(&Stat, MPI_CHAR, &count);
    printf("Task %d: Received %d char(s) from task %d with tag %d \n", rank, count, Stat.MPI_SOURCE, Stat.MPI_TAG);
  }

  elapsed_time = MPI_Wtime() - t1;
  MPI_Finalize();

  if(rank == 0)
    std::cout << "Elapsed time: " << elapsed_time << std::endl;

  return 0;

}
