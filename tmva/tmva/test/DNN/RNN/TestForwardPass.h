// @(#)root/tmva $Id$
// Author: Saurav Shekhar

/*************************************************************************
 * Copyright (C) 2017, Saurav Shekhar
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////
// Generic tests of the RNNLayer Forward pass                     //
////////////////////////////////////////////////////////////////////

#ifndef TMVA_TEST_DNN_TEST_RNN_TEST_FWDPASS_H
#define TMVA_TEST_DNN_TEST_RNN_TEST_FWDPASS_H

#include <iostream>
#include <vector>
#include "../Utility.h"
#include "TMVA/DNN/Functions.h"
#include "TMVA/DNN/RNN/RecurrentNet.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::RNN;

/*! Generate a RecurrentNet, test forward pass */
//______________________________________________________________________________
template <typename Architecture>
auto testForwardPass(size_t timeSteps, size_t batchSize, size_t stateSize, 
                     size_t inputSize)
-> void
{
  using Scalar_t       = typename Architecture::Scalar_t;
  using Matrix_t       = typename Architecture::Matrix_t;
  using Tensor_t       = std::vector<Matrix_t>;
  using RNNLayer_t     = TBasicRNNLayer<Architecture>;  
  using RecurrentNet_t = TRecurrentNet<Architecture>;
 
  std::vector<TMatrixT<Double_t>> XRef;   // T x B x D  
  for (size_t i = 0; i < timeSteps; ++i) {
    randomMatrix(XRef[i]);
  } 
  Tensor_t XArch(XRef);
  
  RNNLayer_t rcell(batchSize, stateSize, inputSize);
  RecurrentNet_t rnn(&rcell, timeSteps);    // passing pointer, take care of lifetime
                                            // maybe use smart pointers
  rnn.Initialize(EInitialization::kGauss);

}

template <typename Architecture>
auto testOneForwardPass(size_t batchSize, size_t stateSize, size_t inputSize)
-> void
{
  using Scalar_t   = typename Architecture::Scalar_t;
  using Matrix_t   = typename Architecture::Matrix_t;
  using RNNLayer_t = TBasicRNNLayer<Architecture>;  
  
  

}

#endif
