// @(#)root/tmva $Id$
// Author: Saurav Shekhar 22/06/17

/*************************************************************************
 * Copyright (C) 2017, Saurav Shekhar                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////
//Testing RNNLayer forward pass for Reference implementation      //
////////////////////////////////////////////////////////////////////

#include <iostream>
#include "TMVA/DNN/Architectures/Reference.h"
#include "TestForwardPass.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::RNN;

int main() {
  std::cout << "Testing RNN Forward pass" << std::endl;

  double error;
  testForwardPass<TReference<double>>(3, 8, 100, 50);

  return 0;
}
