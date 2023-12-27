/***************************************************************************************************
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed,
 * transmitted, transcribed, stored in a retrieval system, or translated into any human
 * or computer language in any form by any means,electronic, mechanical, manual or otherwise,
 * or disclosed to third parties without the express written permission of Samsung Electronics.
 * (Use of the Software is restricted to non-commercial, personal or academic, research purpose
 * only)
 **************************************************************************************************/

#include <bitset>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <random>

#include "Burst.h"
#include "MultiChannelMemorySystem.h"
#include "tests/PIMKernel.h"
#include "tests/TestCases.h"

using namespace std;

void simulateGEMV(shared_ptr<PIMKernel> kernel, uint32_t output_dim, uint32_t input_dim, uint32_t batch_size = 1);
void simulateELT(shared_ptr<PIMKernel> kernel, KernelType kn_type, uint32_t dim, uint32_t batch_size = 1);
void simulateRELU(shared_ptr<PIMKernel> kernel, uint32_t dim);

int main(int argc, char* argv[])
{
    // define PIM kernel
    int numPIMChan = 1;
    int numPIMRank = 1;
    shared_ptr<MultiChannelMemorySystem> mem = make_shared<MultiChannelMemorySystem>(
        "ini/HBM2_samsung_2M_16B_x64.ini", "system_hbm.ini", ".", "example_app",
        256 * numPIMChan * 2);
    shared_ptr<PIMKernel> kernel = make_shared<PIMKernel>(mem, numPIMChan, numPIMRank);

    // define Kernel type and dimension
    simulateGEMV(kernel, 8, 128);
    return 0;

    // testStatsClear();
    // expectAccuracy(KernelType::GEMV, output_dim, dim_data->output_npbst_,
                //    dim_data->getNumElementsPerBlocks());
    return 0;
}

void simulateGEMV(shared_ptr<PIMKernel> kernel, uint32_t output_dim, uint32_t input_dim, uint32_t batch_size) {
    bool is_data = false;
    DataDim *dim_data = new DataDim(KernelType::GEMV, batch_size, output_dim, input_dim, is_data);

    kernel->preloadGemv(&dim_data->weight_npbst_);
    kernel->runPIM();
    cout << "Preload " << kernel->getCycle() << endl;

    kernel->executeGemv(&dim_data->weight_npbst_, &dim_data->input_npbst_, false);
    kernel->runPIM();
    cout << "Execute " << kernel->getCycle() << endl;

    unsigned end_col = kernel->getResultColGemv(
        dim_data->dimTobShape(dim_data->input_dim_), dim_data->output_dim_);

    BurstType* result = new BurstType[dim_data->output_dim_ * dim_data->batch_size_];
    kernel->readResult(result, pimBankType::ODD_BANK,
                        dim_data->output_dim_ * dim_data->batch_size_, 0, 0, end_col);
    kernel->runPIM();
    cout << "Result " << kernel->getCycle() << endl;

    delete[] result;
    delete dim_data;
}

void simulateELT(shared_ptr<PIMKernel> kernel, KernelType kn_type, uint32_t dim, uint32_t batch_size)
{
    bool is_data = false;
    DataDim *dim_data = new DataDim(kn_type, batch_size, dim, dim, is_data);

    int input_row0 = 0;
    int input_row1 = 128;
    int result_row = 256;
    kernel->preloadNoReplacement(&dim_data->input_npbst_, input_row0, 0);
    kernel->preloadNoReplacement(&dim_data->input1_npbst_, input_row1, 0);
    kernel->executeEltwise(dim_data->dimTobShape(dim_data->output_dim_),
                            pimBankType::ALL_BANK, kn_type, input_row0, result_row,
                            input_row1);
    BurstType* result = new BurstType[dim_data->output_dim_];
    kernel->readData(result, dim_data->dimTobShape(dim_data->output_dim_), result_row, 0);
    kernel->runPIM();

    delete[] result;
    delete dim_data;
}

void simulateRELU(shared_ptr<PIMKernel> kernel, uint32_t dim)
{
    // batch == 1?
    bool is_data = false;
    DataDim *dim_data = new DataDim(KernelType::RELU, 1, dim, dim, is_data);

    int input_row0 = 0;
    int result_row = 256;

    kernel->preloadNoReplacement(&dim_data->input_npbst_, input_row0, 0);
    kernel->executeEltwise(dim_data->dimTobShape(dim_data->output_dim_),
                            pimBankType::ALL_BANK, KernelType::RELU, input_row0, result_row);
    BurstType* result = new BurstType[dim_data->output_dim_];
    kernel->readData(result, dim_data->dimTobShape(dim_data->output_dim_), result_row, 0);
    kernel->runPIM();
    cout << "Result " << kernel->getCycle() << endl;

    delete[] result;
    delete dim_data;
}