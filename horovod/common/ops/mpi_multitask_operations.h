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

#ifndef HOROVOD_MPI_MULTITASK_OPERATIONS_H
#define HOROVOD_MPI_MULTITASK_OPERATIONS_H

#define TAG_MULTITASK_ALLREDUCE 2021

#include <iostream>

#include "mpi.h"

#include "collective_operations.h"
#include "../common.h"
#include "../global_state.h"
#include "../mpi/mpi_context.h"

#include <nlohmann/json.hpp>
#include <vector>
#include <memory>
#include <mutex>

#define LINUX
#ifdef LINUX
#include <netinet/in.h>
#include <arpa/inet.h>
#endif

namespace horovod {
namespace common {

class MPI_Multitask_Allreduce : public AllreduceOp {
public:
  MPI_Multitask_Allreduce(MPIContext* mpi_context, HorovodGlobalState* global_state);

  virtual ~MPI_Multitask_Allreduce();

  Status Execute(std::vector<TensorTableEntry>& entries, const Response& response) override;

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

protected:
  MPIContext* mpi_context_;
private:
  int On_Same_Rack(int rank1, int rank2, bool &onSameRack, const nlohmann::json json_data);
  int Step_By_Step_Ring_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
  int Recursive_Doubling_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
  int Copy_Memory_Content(MPI_Datatype type, size_t count, char* pDestBuf, char* pSrcBuf);
  int IP_String_To_Unsigned(const std::string strIP, unsigned int &uIP);
  int IP_Unsigned_To_String(const unsigned int uIP, std::string &strIP);
  int Recv_From_Socket(int sockFd, std::vector<char> &buffer, const unsigned int index, const int maxRealSegSize, unsigned int &size);
  int recv_port_;
  int Send_To_Socket(const char *sendBuf, const unsigned int len, std::string proxyIP, const unsigned int pktIndex, const unsigned int destIP, const unsigned int sourceIP);
  int Copy_Memory_Content_Low_Level(const MPI_Datatype *datatype, int count, char *destination_base, char *source_base);
  std::map<unsigned int, std::vector<char>> reorderBuffer;
  unsigned int packageIndex;
  int recvSoktFd;
  std::mutex soktLock;
  int np, rank, sendTo, recvFrom;
  nlohmann::json json_data;
  bool useProxySend, useProxyRecv;
};

} // namespace common
} // namespace horovod

#endif //HOROVOD_MPI_MULTITASK_OPERATIONS_H
