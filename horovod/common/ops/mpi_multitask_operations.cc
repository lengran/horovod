// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2019 Uber Technologies, Inc.
// Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "mpi_multitask_operations.h"

namespace horovod {
namespace common {

MPI_Multitask_Allreduce::MPI_Multitask_Allreduce(MPIContext* mpi_context, HorovodGlobalState* global_state)
    : packageIndex(0), AllreduceOp(global_state), mpi_context_(mpi_context), recv_port_(2266) 
{
  /* Initialize mpi related variables */
  MPI_Comm comm = mpi_context_->GetMPICommunicator(Communicator::GLOBAL);
  MPI_Comm_size(comm, &np);
  MPI_Comm_rank(comm, &rank);

  /* Initialize receiving-in socket */
  struct sockaddr_in recvServAddr;
  recvSoktFd = socket(AF_INET, SOCK_STREAM, 0);
  if (recvSoktFd == -1)
    throw std::runtime_error("Multitask Allreduce: Create receiving-in socket error(" + std::to_string(errno) + "): " + strerror(errno));
  bzero(&recvServAddr, sizeof(recvServAddr));
  recvServAddr.sin_family = AF_INET;
  recvServAddr.sin_addr.s_addr = htonl(INADDR_ANY);
  recvServAddr.sin_port = htons(recv_port_);
  if (bind(recvSoktFd, (struct sockaddr*)&recvServAddr, sizeof(recvServAddr)) == -1)
    throw std::runtime_error("Bind error(" + std::to_string(errno) + "): " + strerror(errno));
  if (listen(recvSoktFd, (np << 2)))
    throw std::runtime_error("Listen error(" + std::to_string(errno) + "): " + strerror(errno));

  /* Decide which method we use to transmit data, mpi or socket */
  sendTo = (rank + 1) % np;
  recvFrom = (rank + np - 1) % np;
  std::string json_string;
  std::ifstream json_file("look_up_table.json");
  if (!json_file.is_open())
  {
    throw std::runtime_error("Multitask Allreduce: Failed to open loop up table, failed to initialize allreduce operator.");
  }
  // json_file >> json_string;
  std::getline(json_file, json_string);
  json_file.close();
  json_data = nlohmann::json::parse(json_string);

  int ret = On_Same_Rack(sendTo, rank, useProxySend, json_data);
  if (ret != MPI_SUCCESS)
    throw std::runtime_error("Multitask Allreduce: Failed to get rack info.");
  useProxySend = !useProxySend;
  ret = On_Same_Rack(recvFrom, rank, useProxyRecv, json_data);
  if (ret != MPI_SUCCESS)
    throw std::runtime_error("Multitask Allreduce: Failed to get rack info.");
  useProxyRecv = !useProxyRecv;
}
MPI_Multitask_Allreduce::~MPI_Multitask_Allreduce()
{
  close(recvSoktFd);
}

Status MPI_Multitask_Allreduce::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  auto& first_entry = entries[0];

  const void* fused_input_data;
  void* buffer_data;
  size_t buffer_len;
  int64_t num_elements = NumElements(entries);

  // Copy memory into the fusion buffer.
  auto& timeline = global_state_->timeline;
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
    MemcpyInFusionBuffer(entries, fused_input_data, buffer_data, buffer_len);
    timeline.ActivityEndAll(entries);
  } else {
    fused_input_data = first_entry.tensor->data();
    buffer_data = (void*) first_entry.output->data();
    buffer_len = (size_t) first_entry.output->size();
  }

  if (response.prescale_factor() != 1.0) {
    // Execute prescaling op
    ScaleBuffer(response.prescale_factor(), entries, fused_input_data, buffer_data, num_elements);
    fused_input_data = buffer_data; // for unfused, scale is done out of place
  }

  // Do allreduce.
  timeline.ActivityStartAll(entries, MPI_MULTITASK_ALLREDUCE);
  const void* sendbuf = entries.size() > 1 || fused_input_data == buffer_data
                        ? MPI_IN_PLACE : fused_input_data;
  /*int op = MPI_Allreduce(sendbuf, buffer_data,
                         (int) num_elements,
                         mpi_context_->GetMPIDataType(first_entry.tensor),
                         mpi_context_->GetMPISumOp(first_entry.tensor->dtype()),
                         mpi_context_->GetMPICommunicator(Communicator::GLOBAL));*/
  // std::cout << "Calling allreduce function." << std::endl;              // DEBUG
  int op = Step_By_Step_Ring_Allreduce(sendbuf, buffer_data,
                         (int) num_elements,
                         mpi_context_->GetMPIDataType(first_entry.tensor),
                         mpi_context_->GetMPISumOp(first_entry.tensor->dtype()),
                         mpi_context_->GetMPICommunicator(Communicator::GLOBAL));

  if (op != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Allreduce failed, see MPI output for details.");
  }
  // std::cout << "Returned from Allreduce function." << std::endl;        // DEBUG

  timeline.ActivityEndAll(entries);

  if (response.postscale_factor() != 1.0) {
    // Execute postscaling op
    ScaleBuffer(response.postscale_factor(), entries, buffer_data, buffer_data, num_elements);
  }

  // Copy memory out of the fusion buffer.
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
    MemcpyOutFusionBuffer(buffer_data, entries);
    timeline.ActivityEndAll(entries);
  }

  return Status::OK();
}

bool MPI_Multitask_Allreduce::Enabled(const ParameterManager& param_manager,
                           const std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  return true;
}

int MPI_Multitask_Allreduce::Step_By_Step_Ring_Allreduce(const void *sendbuf, void *recvBuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
  int ret;
  // int np, rank;
  // MPI_Comm_size(comm, &np);
  // MPI_Comm_rank(comm, &rank);
  // std::cout << "Start doing allreduce operation." << std::endl;         // DEBUG

  if (np == 1)
    if (sendbuf != MPI_IN_PLACE)
    {
      //memcpy(recvBuf, sendbuf, dataSize);
      ret = Copy_Memory_Content(datatype, count, (char *)recvBuf, (char *)sendbuf);
      if (ret < 0)
        throw std::runtime_error("Multitask Allreduce: np = 1, and we have errors when copy data from srcbuf to dest.");
      return MPI_SUCCESS;
    }
  
  /* If there are less data blocks than workers, use recursive doubling*/
  if (count < np)
  {
    return Recursive_Doubling_Allreduce(sendbuf, recvBuf, count, datatype, op, comm);
  }

  MPI_Aint lb, trueLb, extent, realExtent;
  ret = MPI_Type_get_extent(datatype, &lb, &extent);
  if (ret != MPI_SUCCESS)
    throw std::runtime_error("Multitask Allreduce: Failed to get extent.");
  ret = MPI_Type_get_true_extent(datatype, &trueLb, &realExtent);
  if (ret != MPI_SUCCESS)
    throw std::runtime_error("Multitask Allreduce: Failed to get true extent.");
  
  /* If count can't be devided nicely by np, (count % np) workers will have 
     ([blockSize] + 1) segments of data, and we call these segments "early"
     ones. The others have [blockSize] and we call them "late" ones.
  */
  int lateSegCount = count / np;
  int splitRank = count % np;
  int earlySegCount = lateSegCount;
  if (splitRank != 0)
    earlySegCount = earlySegCount + 1;
  int maxSegCount = earlySegCount;
  int maxRealSegSize = realExtent + (maxSegCount - 1) * extent;

  /* These outputs were used to debug a tricky calculation problem and will never be used. They are here to show how stupid I was */
  // std::cout << "maxRealSegSize = " << std::to_string(maxRealSegSize) << std::endl << "maxSegCount * extent = " << std::to_string(maxSegCount * extent) << std::endl << "extent = " << std::to_string(extent) <<std::endl << "realExtent = " << std::to_string(realExtent) << std::endl;

  std::vector<char> inbuf[2];
  MPI_Request reqs[2];

  inbuf[0].resize(maxRealSegSize);
  if (np > 2)
    inbuf[1].resize(maxRealSegSize);

  /* First we copy the value worker itself contains to receive buffer. It is
     the place where we temporaily store the intermediate value and do the
     calculation.
  */
  if (sendbuf != MPI_IN_PLACE)
  {
    ret = Copy_Memory_Content(datatype, count, (char *)recvBuf, (char *)sendbuf);
    if (ret < 0)
      throw std::runtime_error("Multitask Allreduce: We have errors when copy data from srcbuf to dest.");
  }

  /* Core Computation loop
     
     For each of the remote workers:
     - post irecv for block (r-1)
     - send block (r)
     - in loop for every step k = 2 .. n
     - post irecv for block (r + n - k) % n
     - wait on block (r + n - k + 1) % n to arrive
     - compute on block (r + n - k + 1) % n
     - send block (r + n - k + 1) % n
     - wait on block (r + 1)
     - compute on block (r + 1)
     - send block (r + 1) to rank (r + 1)
     Note that we must be careful when computing the begining of buffers and
     for send operations and computation we must compute the exact block size.
  */

  // int sendTo = (rank + 1) % np;
  // int recvFrom = (rank + np - 1) % np;

  /* First we need to know are these worker on the same rack, by reading a json file */
  // std::string json_string;
  // std::ifstream json_file("look_up_table.json");
  // if (!json_file.is_open())
  // {
  //   return MPI_ERR_FILE;
  // }
  // // json_file >> json_string;
  // std::getline(json_file, json_string);
  // json_file.close();
  // nlohmann::json json_data = nlohmann::json::parse(json_string);

  // bool useProxySend, useProxyRecv;
  // ret = On_Same_Rack(sendTo, rank, useProxySend, json_data);
  // if (ret != MPI_SUCCESS)
  //   throw std::runtime_error("Multitask Allreduce: Failed to get rack info.");
  // useProxySend = !useProxySend;
  // ret = On_Same_Rack(recvFrom, rank, useProxyRecv, json_data);
  // if (ret != MPI_SUCCESS)
  //   throw std::runtime_error("Multitask Allreduce: Failed to get rack info.");
  // useProxyRecv = !useProxyRecv;

  /* Do communication according to their rack-wise relative position */

  /* This is the invertable bit to indicate which data transmission we are currently
     working on out of 2 concurrent transmissions. Its value can be either 0 or 1.
  */
  int inbi = 0;

  int blockOffset, blockCount;
  char *tmpSend, *tmpRecv;

  if (!useProxyRecv && !useProxySend)
  {
    /* Using MPI to send and receive data */
    /* Initialize first receive from the upstream worker. */
    ret = MPI_Irecv(&inbuf[inbi][0], maxSegCount, datatype, recvFrom, TAG_MULTITASK_ALLREDUCE, comm, &reqs[inbi]);
    if (ret != MPI_SUCCESS)
      throw std::runtime_error("Multitask Allreduce: Error for rank " + std::to_string(rank) + " start receiving first block of data from upstream.");
    
    /* Send the first block to the neighbour on the right */
    blockOffset = ((rank < splitRank) ? ((ptrdiff_t)rank * (ptrdiff_t) earlySegCount) : ((ptrdiff_t)rank * (ptrdiff_t)lateSegCount + splitRank));
    blockCount = ((rank < splitRank) ? earlySegCount : lateSegCount);
    tmpSend = ((char *) recvBuf) + blockOffset * extent;
    ret = MPI_Send(tmpSend, blockCount, datatype, sendTo, TAG_MULTITASK_ALLREDUCE, comm);
    if (ret != MPI_SUCCESS)
      throw std::runtime_error("Multitask Allreduce: Error at rank " + std::to_string(rank) + " when sending first block of data.");
    packageIndex++;
    
    /* Send the following blocks of data from the back to the front, each time we send and receive 2 blocks */
    for (int k = 2; k < np; k++)
    {
      int prevBlock = (rank + np - k + 1) % np;

      inbi = inbi ^ 0x1;

      /* Post irecv for the current block */
      ret = MPI_Irecv(&inbuf[inbi][0], maxSegCount, datatype, recvFrom, TAG_MULTITASK_ALLREDUCE, comm, &reqs[inbi]);
      if (ret != MPI_SUCCESS)
        throw std::runtime_error("Multitask Allreduce: Error for rank " + std::to_string(rank) + " to start receiving the " + std::to_string (k) + "th block of data from upstream.");

      /* Wait on previous block to arrive*/
      ret = MPI_Wait(&reqs[inbi ^ 0x1], MPI_STATUS_IGNORE);
      if (ret != MPI_SUCCESS)
        throw std::runtime_error("Multitask Allreduce: Error for rank " + std::to_string(rank) + " to get the " + std::to_string (k) + "th block of data from upstream.");

      /* Apply operation on previous block: result goes to recvBuf*/
      blockOffset = ((prevBlock < splitRank) ? ((ptrdiff_t)prevBlock * earlySegCount) : ((ptrdiff_t)prevBlock * lateSegCount + splitRank));
      blockCount = ((prevBlock < splitRank) ? earlySegCount : lateSegCount);
      tmpRecv = ((char *)recvBuf) + (ptrdiff_t)blockOffset * extent;
      MPI_Reduce_local(&inbuf[inbi ^ 0x1][0], tmpRecv, blockCount, datatype, op);

      /* Send previous block to next neighbour*/
      ret = MPI_Send(tmpRecv, blockCount, datatype, sendTo, TAG_MULTITASK_ALLREDUCE, comm);
      if (ret != MPI_SUCCESS)
        throw std::runtime_error("Multitask Allreduce: Error for rank " + std::to_string(rank) + " to send the " + std::to_string (k) + "th block of data to downstream.");
      packageIndex++;
    }

    /* Wait for the last block to arrive */
    ret = MPI_Wait(&reqs[inbi], MPI_STATUS_IGNORE);
    if (ret != MPI_SUCCESS)
      throw std::runtime_error("Multitask Allreduce: Error for rank " + std::to_string(rank) + " to receive the last block of data from upstream.");
    
    /* Apply operation on the last block */
    int lastBlock = (rank + 1) % np;
    blockOffset = ((lastBlock < splitRank) ? ((ptrdiff_t)lastBlock * earlySegCount) : ((ptrdiff_t)lastBlock * lateSegCount + splitRank));
    blockCount = ((lastBlock < splitRank) ? earlySegCount : lateSegCount);
    tmpRecv = ((char *)recvBuf) + (ptrdiff_t)blockOffset * extent;
    MPI_Reduce_local(&inbuf[inbi][0], tmpRecv, blockCount, datatype, op);

    /* Distribution loop - variation of ring allgather */
    for (int k = 0; k < np - 1; k++)
    {
      int recvDataBlock = (rank + np - k) % np;
      int sendDataBlock = (rank + np - k + 1) % np;
      int sendBlockOffset = ((sendDataBlock < splitRank) ? ((ptrdiff_t)sendDataBlock * earlySegCount) : ((ptrdiff_t)sendDataBlock * lateSegCount + splitRank));
      int recvBlockOffset = ((recvDataBlock < splitRank) ? ((ptrdiff_t)recvDataBlock * earlySegCount) : ((ptrdiff_t)recvDataBlock * lateSegCount + splitRank));
      blockCount = ((sendDataBlock < splitRank) ? earlySegCount : lateSegCount);

      tmpRecv = (char *)recvBuf + (ptrdiff_t)recvBlockOffset * extent;
      tmpSend = (char *)recvBuf + (ptrdiff_t)sendBlockOffset * extent;

      ret = MPI_Sendrecv(tmpSend, blockCount, datatype, sendTo, TAG_MULTITASK_ALLREDUCE, tmpRecv, maxSegCount, datatype, recvFrom, TAG_MULTITASK_ALLREDUCE, comm, MPI_STATUS_IGNORE);
      if (ret != MPI_SUCCESS)
        throw std::runtime_error("Multitask Allreduce: Error for rank " + std::to_string(rank) + " to send & receive the " + std::to_string (k) + "th block of data.");
      packageIndex++;
    }
  }
  else if (useProxyRecv && useProxySend)
  {
    /* Using socket to asynchronmously receive and send data via proxy server */
    // /* Initialize receiving-in socket */
    // struct sockaddr_in recvServAddr;
    // int recvSoktFd = socket(AF_INET, SOCK_STREAM, 0);
    // if (recvSoktFd == -1)
    //   throw std::runtime_error("Multitask Allreduce: Create receiving-in socket error(" + std::to_string(errno) + "): " + strerror(errno));
    // bzero(&recvServAddr, sizeof(recvServAddr));
    // recvServAddr.sin_family = AF_INET;
    // recvServAddr.sin_addr.s_addr = htonl(INADDR_ANY);
    // recvServAddr.sin_port = htons(recv_port_);
    // if (bind(recvSoktFd, (struct sockaddr*)&recvServAddr, sizeof(recvServAddr)) == -1)
    //   throw std::runtime_error("Bind error(" + std::to_string(errno) + "): " + strerror(errno));
    // if (listen(recvSoktFd, (np << 2)))
    //   throw std::runtime_error("Listen error(" + std::to_string(errno) + "): " + strerror(errno));
    
    /* Initialize out-going socket (just the IP here, to asure each message send by a seprate) */
    std::string proxyIP = json_data["rank_to_ip"][std::to_string(rank)];
    unsigned int iSourceIP;
    IP_String_To_Unsigned(proxyIP, iSourceIP);
    proxyIP = json_data["ip_to_rack"][proxyIP];
    std::string sNeighbourIP = json_data["rank_to_ip"][std::to_string(sendTo)];
    unsigned int iNeighbourIP;
    IP_String_To_Unsigned(sNeighbourIP, iNeighbourIP);

    /* Initialize first receive from the upstream worker. */
    unsigned int msgSize;
    std::vector<std::thread> recvThreads(2);
    recvThreads[0] = std::thread(&MPI_Multitask_Allreduce::Recv_From_Socket, this, recvSoktFd, std::ref(inbuf[inbi]), packageIndex, maxRealSegSize, std::ref(msgSize));
    
    /* Send first block (local block) to the neighbour */
    blockOffset = ((rank < splitRank) ? ((ptrdiff_t)rank * (ptrdiff_t) earlySegCount) : ((ptrdiff_t)rank * (ptrdiff_t)lateSegCount + splitRank));
    blockCount = ((rank < splitRank) ? earlySegCount : lateSegCount);
    tmpSend = ((char *) recvBuf) + blockOffset * extent;
    ret = Send_To_Socket(tmpSend, (extent * blockCount), proxyIP, packageIndex, iNeighbourIP, iSourceIP);
    if (ret != 0)
      throw std::runtime_error("Multitask Allreduce: Error at rank " + std::to_string(rank) + " when sending first block of data.");
    packageIndex++;
    
    /* Send the following blocks of data from the back to the front, each time we send and receive 2 blocks */
    for (int k = 2; k < np; k++)
    {
      // std::cout << "k = " << std::to_string(k) << ", np = " << std::to_string(np) << std::endl;
      int prevBlock = (rank + np - k + 1) % np;

      inbi = inbi ^ 0x1;

      /* Post asynchronous receive for current block */
      recvThreads[inbi] = std::thread(&MPI_Multitask_Allreduce::Recv_From_Socket, this, recvSoktFd, std::ref(inbuf[inbi]), packageIndex, maxRealSegSize, std::ref(msgSize));
      
      /* Wait on previous block to arrive */
      recvThreads[inbi ^ 0x1].join();

      /* Apply operation on previous block (result goes to recvBuf) */
      blockOffset = ((prevBlock < splitRank) ? ((ptrdiff_t)prevBlock * earlySegCount) : ((ptrdiff_t)prevBlock * lateSegCount + splitRank));
      blockCount = ((prevBlock < splitRank) ? earlySegCount : lateSegCount);
      tmpRecv = ((char *)recvBuf) + (ptrdiff_t)blockOffset * extent;
      MPI_Reduce_local(&inbuf[inbi ^ 0x1][0], tmpRecv, blockCount, datatype, op);

      /* Send previous block to the send-to worker*/
      ret = Send_To_Socket(tmpRecv, (extent * blockCount), proxyIP, packageIndex, iNeighbourIP, iSourceIP);
      if (ret != 0)
        throw std::runtime_error("Multitask Allreduce: Error at rank " + std::to_string(rank) + " while sending the " + std::to_string(k) + "th block of data.");
      packageIndex++;
    }

    /* Wait on the last block to arrive */
    recvThreads[inbi].join();
    // std::cout << "Received the last block of data." << std::endl;      // DEBUG

    /* Apply operation on the last block */
    int lastBlock = (rank + 1) % np;
    blockOffset = ((lastBlock < splitRank) ? ((ptrdiff_t)lastBlock * earlySegCount) : ((ptrdiff_t)lastBlock * lateSegCount + splitRank));
    blockCount = ((lastBlock < splitRank) ? earlySegCount : lateSegCount);
    tmpRecv = ((char *)recvBuf) + (ptrdiff_t)blockOffset * extent;
    MPI_Reduce_local(&inbuf[inbi][0], tmpRecv, blockCount, datatype, op);


    /* Distribution loop - variation of ring allgather */
    for (int k = 0; k < np - 1; k++)
    {
      int recvDataBlock = (rank + np - k) % np;
      int sendDataBlock = (rank + np - k + 1) % np;
      int sendBlockOffset = ((sendDataBlock < splitRank) ? ((ptrdiff_t)sendDataBlock * earlySegCount) : ((ptrdiff_t)sendDataBlock * lateSegCount + splitRank));
      int recvBlockOffset = ((recvDataBlock < splitRank) ? ((ptrdiff_t)recvDataBlock * earlySegCount) : ((ptrdiff_t)recvDataBlock * lateSegCount + splitRank));
      blockCount = ((sendDataBlock < splitRank) ? earlySegCount : lateSegCount);

      tmpRecv = (char *)recvBuf + (ptrdiff_t)recvBlockOffset * extent;
      tmpSend = (char *)recvBuf + (ptrdiff_t)sendBlockOffset * extent;

      // std::cout << "Start receiving the second message." << std::endl;                   // DEBUG
      recvThreads[0] = std::thread(&MPI_Multitask_Allreduce::Recv_From_Socket, this, recvSoktFd, std::ref(inbuf[0]), packageIndex, maxRealSegSize, std::ref(msgSize));
      // std::cout << "Start sending the second message." << std::endl;                   // DEBUG
      ret = Send_To_Socket(tmpSend, (blockCount * extent), proxyIP, packageIndex, iNeighbourIP, iSourceIP);
      if (ret != 0)
        throw std::runtime_error("Multitask Allreduce: Error at rank " + std::to_string(rank) + " while sending the " + std::to_string(k) + "th final block of data.");
      recvThreads[0].join();
      memcpy(tmpRecv, &inbuf[0][0], (maxSegCount * extent));
      packageIndex++;
    }

    // close(recvSoktFd);
  }
  else if (!useProxyRecv && useProxySend)
  {
    /* Using MPI to receive data but using socket to send data */
    /* Initialize out-going socket (just the IP here, to asure each message send by a seprate) */
    std::string proxyIP = json_data["rank_to_ip"][std::to_string(rank)];
    unsigned int iSourceIP;
    IP_String_To_Unsigned(proxyIP, iSourceIP);
    proxyIP = json_data["ip_to_rack"][proxyIP];
    std::string sNeighbourIP = json_data["rank_to_ip"][std::to_string(sendTo)];
    unsigned int iNeighbourIP;
    IP_String_To_Unsigned(sNeighbourIP, iNeighbourIP);

    /* Initialize first receive from the upstream worker. */
    ret = MPI_Irecv(&inbuf[inbi][0], maxSegCount, datatype, recvFrom, TAG_MULTITASK_ALLREDUCE, comm, &reqs[inbi]);
    if (ret != MPI_SUCCESS)
      throw std::runtime_error("Multitask Allreduce: Error for rank " + std::to_string(rank) + " start receiving first block of data from upstream.");

    /* Send first block (local block) to the neighbour */
    blockOffset = ((rank < splitRank) ? ((ptrdiff_t)rank * (ptrdiff_t) earlySegCount) : ((ptrdiff_t)rank * (ptrdiff_t)lateSegCount + splitRank));
    blockCount = ((rank < splitRank) ? earlySegCount : lateSegCount);
    tmpSend = ((char *) recvBuf) + blockOffset * extent;
    ret = Send_To_Socket(tmpSend, (extent * blockCount), proxyIP, packageIndex, iNeighbourIP, iSourceIP);
    if (ret != 0)
      throw std::runtime_error("Multitask Allreduce: Error at rank " + std::to_string(rank) + " when sending first block of data.");
    packageIndex++;
    
    /* Send the following blocks of data from the back to the front, each time we send and receive 2 blocks */
    for (int k = 2; k < np; k++)
    {
      int prevBlock = (rank + np - k + 1) % np;

      inbi = inbi ^ 0x1;

      /* Post irecv for the current block */
      ret = MPI_Irecv(&inbuf[inbi][0], maxSegCount, datatype, recvFrom, TAG_MULTITASK_ALLREDUCE, comm, &reqs[inbi]);
      if (ret != MPI_SUCCESS)
        throw std::runtime_error("Multitask Allreduce: Error for rank " + std::to_string(rank) + " to start receiving the " + std::to_string (k) + "th block of data from upstream.");

      /* Wait on previous block to arrive*/
      ret = MPI_Wait(&reqs[inbi ^ 0x1], MPI_STATUS_IGNORE);
      if (ret != MPI_SUCCESS)
        throw std::runtime_error("Multitask Allreduce: Error for rank " + std::to_string(rank) + " to get the " + std::to_string (k) + "th block of data from upstream.");
      
      /* Apply operation on previous block: result goes to recvBuf*/
      blockOffset = ((prevBlock < splitRank) ? ((ptrdiff_t)prevBlock * earlySegCount) : ((ptrdiff_t)prevBlock * lateSegCount + splitRank));
      blockCount = ((prevBlock < splitRank) ? earlySegCount : lateSegCount);
      tmpRecv = ((char *)recvBuf) + (ptrdiff_t)blockOffset * extent;
      MPI_Reduce_local(&inbuf[inbi ^ 0x1][0], tmpRecv, blockCount, datatype, op);

      /* Send previous block to the send-to worker*/
      ret = Send_To_Socket(tmpRecv, (extent * blockCount), proxyIP, packageIndex, iNeighbourIP, iSourceIP);
      if (ret != 0)
        throw std::runtime_error("Multitask Allreduce: Error at rank " + std::to_string(rank) + " while sending the " + std::to_string(k) + "th block of data.");
      packageIndex++;
    }

    /* Wait for the last block to arrive */
    ret = MPI_Wait(&reqs[inbi], MPI_STATUS_IGNORE);
    if (ret != MPI_SUCCESS)
      throw std::runtime_error("Multitask Allreduce: Error for rank " + std::to_string(rank) + " to receive the last block of data from upstream.");
    
    /* Apply operation on the last block */
    int lastBlock = (rank + 1) % np;
    blockOffset = ((lastBlock < splitRank) ? ((ptrdiff_t)lastBlock * earlySegCount) : ((ptrdiff_t)lastBlock * lateSegCount + splitRank));
    blockCount = ((lastBlock < splitRank) ? earlySegCount : lateSegCount);
    tmpRecv = ((char *)recvBuf) + (ptrdiff_t)blockOffset * extent;
    MPI_Reduce_local(&inbuf[inbi][0], tmpRecv, blockCount, datatype, op);

    /* Distribution loop - variation of ring allgather */
    for (int k = 0; k < np - 1; k++)
    {
      int recvDataBlock = (rank + np - k) % np;
      int sendDataBlock = (rank + np - k + 1) % np;
      int sendBlockOffset = ((sendDataBlock < splitRank) ? ((ptrdiff_t)sendDataBlock * earlySegCount) : ((ptrdiff_t)sendDataBlock * lateSegCount + splitRank));
      int recvBlockOffset = ((recvDataBlock < splitRank) ? ((ptrdiff_t)recvDataBlock * earlySegCount) : ((ptrdiff_t)recvDataBlock * lateSegCount + splitRank));
      blockCount = ((sendDataBlock < splitRank) ? earlySegCount : lateSegCount);

      tmpRecv = (char *)recvBuf + (ptrdiff_t)recvBlockOffset * extent;
      tmpSend = (char *)recvBuf + (ptrdiff_t)sendBlockOffset * extent;

      ret = MPI_Irecv(tmpRecv, maxSegCount, datatype, recvFrom, TAG_MULTITASK_ALLREDUCE, comm, &reqs[0]); 
      if (ret != MPI_SUCCESS)
        throw std::runtime_error("Multitask Allreduce: Error for rank " + std::to_string(rank) + " to start receiving the " + std::to_string (k) + "th block of final data.");
      ret = Send_To_Socket(tmpSend, (blockCount * extent), proxyIP, packageIndex, iNeighbourIP, iSourceIP);
      if (ret != 0)
        throw std::runtime_error("Multitask Allreduce: Error at rank " + std::to_string(rank) + " while sending the " + std::to_string(k) + "th final block of data.");
      ret = MPI_Wait(&reqs[0], MPI_STATUS_IGNORE);
      if (ret != MPI_SUCCESS)
        throw std::runtime_error("Multitask Allreduce: Error for rank " + std::to_string(rank) + " to receive the " + std::to_string (k) + "th block of final data.");
      packageIndex++;
    }
  }
  else
  {
    /* Using socket to receive data but using MPI to send data */
    // /* Initialize receiving-in socket */
    // struct sockaddr_in recvServAddr;
    // int recvSoktFd = socket(AF_INET, SOCK_STREAM, 0);
    // if (recvSoktFd == -1)
    //   throw std::runtime_error("Multitask Allreduce: Create receiving-in socket error(" + std::to_string(errno) + "): " + strerror(errno));
    // bzero(&recvServAddr, sizeof(recvServAddr));
    // recvServAddr.sin_family = AF_INET;
    // recvServAddr.sin_addr.s_addr = htonl(INADDR_ANY);
    // recvServAddr.sin_port = htons(recv_port_);
    // if (bind(recvSoktFd, (struct sockaddr*)&recvServAddr, sizeof(recvServAddr)) == -1)
    //   throw std::runtime_error("Bind error(" + std::to_string(errno) + "): " + strerror(errno));
    // if (listen(recvSoktFd, (np << 2)))
    //   throw std::runtime_error("Listen error(" + std::to_string(errno) + "): " + strerror(errno));
    
    /* Initialize first receive from the upstream worker. */
    unsigned int msgSize;
    std::vector<std::thread> recvThreads(2);
    recvThreads[0] = std::thread(&MPI_Multitask_Allreduce::Recv_From_Socket, this, recvSoktFd, std::ref(inbuf[inbi]), packageIndex, maxRealSegSize, std::ref(msgSize));
    
    /* Send the first block to the neighbour on the right */
    blockOffset = ((rank < splitRank) ? ((ptrdiff_t)rank * (ptrdiff_t) earlySegCount) : ((ptrdiff_t)rank * (ptrdiff_t)lateSegCount + splitRank));
    blockCount = ((rank < splitRank) ? earlySegCount : lateSegCount);
    tmpSend = ((char *) recvBuf) + blockOffset * extent;
    ret = MPI_Send(tmpSend, blockCount, datatype, sendTo, TAG_MULTITASK_ALLREDUCE, comm);
    if (ret != MPI_SUCCESS)
      throw std::runtime_error("Multitask Allreduce: Error at rank " + std::to_string(rank) + " when sending first block of data.");
    packageIndex++;

    /* Send the following blocks of data from the back to the front, each time we send and receive 2 blocks */
    for (int k = 2; k < np; k++)
    {
      int prevBlock = (rank + np - k + 1) % np;

      inbi = inbi ^ 0x1;

      /* Post asynchronous receive for current block */
      recvThreads[inbi] = std::thread(&MPI_Multitask_Allreduce::Recv_From_Socket, this, recvSoktFd, std::ref(inbuf[inbi]), packageIndex, maxRealSegSize, std::ref(msgSize));
      
      /* Wait on previous block to arrive */
      recvThreads[inbi ^ 0x1].join();

      /* Apply operation on previous block (result goes to recvBuf) */
      blockOffset = ((prevBlock < splitRank) ? ((ptrdiff_t)prevBlock * earlySegCount) : ((ptrdiff_t)prevBlock * lateSegCount + splitRank));
      blockCount = ((prevBlock < splitRank) ? earlySegCount : lateSegCount);
      tmpRecv = ((char *)recvBuf) + (ptrdiff_t)blockOffset * extent;
      MPI_Reduce_local(&inbuf[inbi ^ 0x1][0], tmpRecv, blockCount, datatype, op);

      /* Send previous block to next neighbour*/
      ret = MPI_Send(tmpRecv, blockCount, datatype, sendTo, TAG_MULTITASK_ALLREDUCE, comm);
      if (ret != MPI_SUCCESS)
        throw std::runtime_error("Multitask Allreduce: Error for rank " + std::to_string(rank) + " to send the " + std::to_string (k) + "th block of data to downstream.");
      packageIndex++;
    }

    /* Wait on the last block to arrive */
    recvThreads[inbi].join();

    /* Apply operation on the last block */
    int lastBlock = (rank + 1) % np;
    blockOffset = ((lastBlock < splitRank) ? ((ptrdiff_t)lastBlock * earlySegCount) : ((ptrdiff_t)lastBlock * lateSegCount + splitRank));
    blockCount = ((lastBlock < splitRank) ? earlySegCount : lateSegCount);
    tmpRecv = ((char *)recvBuf) + (ptrdiff_t)blockOffset * extent;
    MPI_Reduce_local(&inbuf[inbi][0], tmpRecv, blockCount, datatype, op);

    /* Distribution loop - variation of ring allgather */
    for (int k = 0; k < np - 1; k++)
    {
      int recvDataBlock = (rank + np - k) % np;
      int sendDataBlock = (rank + np - k + 1) % np;
      int sendBlockOffset = ((sendDataBlock < splitRank) ? ((ptrdiff_t)sendDataBlock * earlySegCount) : ((ptrdiff_t)sendDataBlock * lateSegCount + splitRank));
      int recvBlockOffset = ((recvDataBlock < splitRank) ? ((ptrdiff_t)recvDataBlock * earlySegCount) : ((ptrdiff_t)recvDataBlock * lateSegCount + splitRank));
      blockCount = ((sendDataBlock < splitRank) ? earlySegCount : lateSegCount);

      tmpRecv = (char *)recvBuf + (ptrdiff_t)recvBlockOffset * extent;
      tmpSend = (char *)recvBuf + (ptrdiff_t)sendBlockOffset * extent;

      recvThreads[0] = std::thread(&MPI_Multitask_Allreduce::Recv_From_Socket, this, recvSoktFd, std::ref(inbuf[0]), packageIndex, maxRealSegSize, std::ref(msgSize));
      ret = MPI_Send(tmpSend, blockCount, datatype, sendTo, TAG_MULTITASK_ALLREDUCE, comm);
      if (ret != MPI_SUCCESS)
        throw std::runtime_error("Multitask Allreduce: Error for rank " + std::to_string(rank) + " to send the " + std::to_string (k) + "th block of final data.");
      recvThreads[0].join();
      memcpy(tmpRecv, &inbuf[0][0], (maxSegCount * extent));
      packageIndex++;
    }

    // close(recvSoktFd);
  }
  
  // std::cout << "Multitask Allreduce: We are bloody done!" << std::endl;
  return MPI_SUCCESS;
}

int MPI_Multitask_Allreduce::Recv_From_Socket(int sockFd, std::vector<char> &buffer, const unsigned int index, const int maxRealSegSize, unsigned int &size)
{
  bool stop = false;
  while (!stop)
  {
    /* First check reorder buffer */
    std::map<unsigned int, std::vector<char>>::iterator it = reorderBuffer.find(index);
    if (it != reorderBuffer.end())
    {
      /* The package has been received before, just copy it to the buffer */
      size = *(unsigned int *)&(it->second[8]);
      memcpy(&buffer[0], &(it->second[16]), size);

      reorderBuffer.erase(it->first);
      stop = true;
      break;
    }

    /* Receive package */
    soktLock.lock();
    // std::cout << "Socket locked." << std::endl;                   // DEBUG
    int connFd = accept(recvSoktFd, NULL, NULL);
    if (connFd == -1)
      throw std::runtime_error("Multitask Allreduce: Accept " + std::to_string(recvSoktFd) + " error(" + std::to_string(errno) + "): " + strerror(errno));
    // std::cout << "Accept from socket." << std::endl;                   // DEBUG

    /* First receive header: src, dest, packet size, index */
    std::vector<char> header(16);
    bzero(&header[0], 16);
    recv(connFd, &header[0], 16, 0);
    unsigned int sourceID, destID, indexReveived;
    sourceID = *(unsigned int *)(&header[0]);
    destID = *(unsigned int *)(&header[4]);
    size = *(unsigned int *)(&header[8]);
    indexReveived = *(unsigned int *)(&header[12]);
    // std::cout << "Header: SourceID " << sourceID << ", Destination ID " << destID << ", size " << size << ", index received " << indexReveived << ", index expected " << index << ",size expect " << maxRealSegSize << "/" << buffer.size() << std::endl;        // DEBUG

    /* Check package index, if it's not what we want, put it into the reorder buffer */
    if (indexReveived != index)
    {
      std::vector<char> storedPackage(16 + size);
      memcpy(&storedPackage[0], &sourceID, 4);
      memcpy(&storedPackage[4], &destID, 4);
      memcpy(&storedPackage[8], &size, 4);
      memcpy(&storedPackage[12], &indexReveived, 4);
      
      /* Receive data */
      bzero(&storedPackage[16], size);
      // std::cout << "Start receiving message body to reorder buffer." << std::endl;      // DEBUG
      int received = recv(connFd, &storedPackage[16], size, 0);
      int receivedAll = received;
      // std::cout << "Message in reorder buffer received: " << (receivedAll * 100 / size) << "%" << std::endl;      // DEBUG
      while (receivedAll < size)
      {
        received = recv(connFd, &storedPackage[receivedAll + 16], size - receivedAll, 0);
        receivedAll = receivedAll + received;
      }

      /* Save package to reorder buffer. */
      reorderBuffer[indexReveived] = storedPackage;
      // std::cout << "Message saved in reorder buffer." << std::endl;                   // DEBUG
    }
    else
    {
      /* It it the package we expected. Receive actual data */
      bzero(&buffer[0], size);
      // std::cout << "Receive message into buffer." << std::endl;                   // DEBUG
      int received = recv(connFd, &buffer[0], size, 0);
      int receivedAll = received;
      // std::cout << "Message received: " << (receivedAll * 100 / size) << "%" << std::endl;      // DEBUG
      while (receivedAll < size)
      {
        received = recv(connFd, &buffer[receivedAll], size - receivedAll, 0);
        receivedAll = receivedAll + received;
      }
      // std::cout << "Message received." << std::endl;      // DEBUG

      stop = true;
    }
    
    close(connFd);
    soktLock.unlock();
    // std::cout << "Socket lock released." << std::endl;
  }

  return 0;
}

int MPI_Multitask_Allreduce::Send_To_Socket(const char *sendBuf, const unsigned int len, std::string proxyIP, const unsigned int pktIndex, const unsigned int destIP, const unsigned int sourceIP)
{
  /* Initialization */
  struct sockaddr_in sendServAddr;
  int sendSoktFd = socket(AF_INET, SOCK_STREAM, 0);
  if (sendSoktFd == -1)
    throw std::runtime_error("Multitask Allreduce: Create sending-out socket error(" + std::to_string(errno) + "): " + strerror(errno));
  bzero(&sendServAddr, sizeof(sendServAddr));
  sendServAddr.sin_family = AF_INET;
  inet_pton(AF_INET, proxyIP.data(), &sendServAddr.sin_addr);
  sendServAddr.sin_port = htons(3366);

  if (connect(sendSoktFd, (struct sockaddr*)&sendServAddr, sizeof(sendServAddr)) == -1)
    throw std::runtime_error("Multitask Allreduce: Connection error(" + std::to_string(errno) + "): " + strerror(errno));
  
  /* Assmble and send header */
  char header[16];
  memcpy(&header[0], &sourceIP, 4);
  memcpy(&header[4], &destIP, 4);
  memcpy(&header[8], &len, 4);
  memcpy(&header[12], &pktIndex, 4);
  size_t sent = send(sendSoktFd, header, 16, 0);
  if (sent != 16)
    throw std::runtime_error("Multitask Allreduce: Failed to send package header.");

  /* Send actual data */
  sent = send(sendSoktFd, sendBuf, len, 0);
  while (sent < len)
  {
    sent += send(sendSoktFd, sendBuf + sent, len - sent, 0);
  }

  close(sendSoktFd);
  return 0;
}

int MPI_Multitask_Allreduce::IP_String_To_Unsigned(const std::string strIP, unsigned int &uIP)
{
  /* First replace all dot by space */
  size_t fstDot = strIP.find('.');
  size_t sndDot = strIP.find('.', fstDot + 1);
  size_t trdDot = strIP.find('.', sndDot + 1);

  std::string strIPWithSpace(strIP);
  strIPWithSpace[fstDot] = ' ';
  strIPWithSpace[sndDot] = ' ';
  strIPWithSpace[trdDot] = ' ';

  std::stringstream stream(strIPWithSpace);
  unsigned int tmp1, tmp2, tmp3, tmp4;            // 4 parts of IP from left to right
  stream >> tmp1 >> tmp2 >> tmp3 >> tmp4;
  uIP = tmp4 | (tmp3 << 8) | (tmp2 << 16) | (tmp1 << 24);

  return 0;
}

int MPI_Multitask_Allreduce::IP_Unsigned_To_String(const unsigned int uIP, std::string &strIP)
{
  unsigned int tmp1 = (uIP & 0xFF000000) >> 24;
  unsigned int tmp2 = (uIP & 0x00FF0000) >> 16;
  unsigned int tmp3 = (uIP & 0x0000FF00) >> 8;
  unsigned int tmp4 = uIP & 0x000000FF;

  strIP = std::to_string(tmp1) + "." + std::to_string(tmp2) + "." + std::to_string(tmp3) + "." + std::to_string(tmp4);

  return 0;
}

int MPI_Multitask_Allreduce::On_Same_Rack(int rank1, int rank2, bool &onSameRack, const nlohmann::json json_data)
{
  std::string ip1 = json_data["rank_to_ip"][std::to_string(rank1)];
  std::string ip2 = json_data["rank_to_ip"][std::to_string(rank2)];

  std::string rack1 = json_data["ip_to_rack"][ip1];
  std::string rack2 = json_data["ip_to_rack"][ip2];

  onSameRack = rack1.compare(rack2) == 0;

  return MPI_SUCCESS;
}

int MPI_Multitask_Allreduce::Recursive_Doubling_Allreduce(const void *sendbuf, void *recvBuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
  // Maybe we won't use this if lucky, but still TO DO
  return MPI_SUCCESS;
}

int MPI_Multitask_Allreduce::Copy_Memory_Content(MPI_Datatype type, size_t count, char* pDestBuf, char* pSrcBuf)
{
  /* YOLO */
    int size;
    MPI_Type_size(type, &size);
    memcpy(pDestBuf, pSrcBuf, size * count);
    
    return MPI_SUCCESS;

  int length, rc;
  ptrdiff_t lb, extent;

  MPI_Type_get_extent(type, &lb, &extent);
  while (count != 0)
  {
    length = INT_MAX;
    if (((size_t)length) > count)
      length = count;
    // rc = Copy_Memory_Content_Low_Level((&type)->super, length, pDestBuf, pSrcBuf);
    if (rc != 0)
      return rc;
    pDestBuf += ((ptrdiff_t)length) * extent;
    pSrcBuf += ((ptrdiff_t)length) * extent;
    count -= (size_t)length;
  }

  return MPI_SUCCESS;
}

int MPI_Multitask_Allreduce::Copy_Memory_Content_Low_Level(const MPI_Datatype *datatype, int count, char *destination_base, char *source_base)
{
  ptrdiff_t extent;
  int (*fct)(const MPI_Datatype*, int, char*, char*);


}

} // namespace common
} // namespace horovod
