#include "miopen.hpp"
#include "tensor.hpp"

int main(int argc, char *argv[]) {
  // Regular MIOpen housekeeping
  device_init();
  //miopenEnableProfiling(mio::instance()->handle(), true);
  //Tensor (n,c,h,w)
  Tensor input1(1, 64, 56, 56); // batch size = 1, input channels = 64, image size = 56 x 56
  Tensor output1(1, 64, 56, 56);// 1, 64, 56, 56
  Tensor weights1(64, 64, 1, 1); // 64 kernel of size 64 x 1 x 1
  Tensor bias1(1, 64, 1, 1);//   1, 64, 1, 1

  //Tensor input2( 1, 64, 56, 56); // batch size = 1, input channels = 64, image size = 56 x 56
  Tensor output2(1, 64, 56, 56); // 1, 64, 56, 56
  Tensor weights2(64, 64, 3, 3); // 64 kernel of size 64 x 3 x 3
  Tensor bias2(1, 64, 1, 1);//   1, 64, 1, 1

  miopenConvolutionDescriptor_t conv_desc1, conv_desc2;

  // initialize tensor
  input1.uniform();
  weights1.uniform();
  // declarations for fusion
  miopenFusionPlanDescriptor_t fusePlanDesc1, fusePlanDesc2;
  miopenOperatorArgs_t fusionArgs1, fusionArgs2;
  miopenFusionOpDescriptor_t convoOp1, convoOp2;
  miopenFusionOpDescriptor_t biasOp1, biasOp2;
  miopenFusionOpDescriptor_t activOp1, activOp2;

  {
	  // Create the convolution descriptor
	  miopenCreateConvolutionDescriptor(&conv_desc1);
	  miopenInitConvolutionDescriptor(conv_desc1, miopenConvolution, 0, 0, 1, 1, 1, 1);
	  // Get the convolution output dimensions
	  int n, c, h, w;
	  miopenGetConvolutionForwardOutputDim(conv_desc1, input1.desc, weights1.desc, &n, &c, &h, &w);
	  assert(n==1 && c == 64 && h ==56 && w ==56 );// Just making sure
	  // Create the fusion plan
	  miopenCreateFusionPlan(&fusePlanDesc1, miopenVerticalFusion, input1.desc);
	  miopenCreateOperatorArgs(&fusionArgs1);
	  miopenCreateOpConvForward(fusePlanDesc1, &convoOp1, conv_desc1, weights1.desc);
	  miopenCreateOpBiasForward(fusePlanDesc1, &biasOp1, bias1.desc);
	  // we are only concerned with RELU
	  miopenCreateOpActivationForward(fusePlanDesc1, &activOp1, miopenActivationRELU);

	  // compile fusion plan
	  auto status = miopenCompileFusionPlan(mio::instance()->handle(), fusePlanDesc1);
	  if (status != miopenStatusSuccess) {
		return -1;
	  }
	  float alpha = static_cast<float>(1), beta = static_cast<float>(0);
	  float activ_alpha = static_cast<float>(1), activ_beta = static_cast<float>(0), activ_gamma = static_cast<float>(1);

	  // Set the Args
	  miopenSetOpArgsConvForward(fusionArgs1, convoOp1, &alpha, &beta, weights1.data);
	  miopenSetOpArgsActivForward(fusionArgs1, activOp1, &alpha, &beta, activ_alpha, activ_beta, activ_gamma);
	  miopenSetOpArgsBiasForward(fusionArgs1, biasOp1, &alpha, &beta, bias1.data);
	  input1.uniform();
	  weights1.uniform();
  }

  {
	  // Create the convolution descriptor
	  miopenCreateConvolutionDescriptor(&conv_desc2);
	  miopenInitConvolutionDescriptor(conv_desc2, miopenConvolution, 1, 1, 1, 1, 1, 1);
	  // Get the convolution output dimensions
	  int n, c, h, w;
	  miopenGetConvolutionForwardOutputDim(conv_desc2, output1.desc, weights2.desc, &n, &c, &h, &w);
	  assert(n==1 && c == 64 && h ==56 && w ==56 );// Just making sure
	  // Create the fusion plan
	  miopenCreateFusionPlan(&fusePlanDesc2, miopenVerticalFusion, output1.desc);
	  miopenCreateOperatorArgs(&fusionArgs2);
	  miopenCreateOpConvForward(fusePlanDesc2, &convoOp2, conv_desc2, weights2.desc);
	  miopenCreateOpBiasForward(fusePlanDesc2, &biasOp2, bias2.desc);
	  // we are only concerned with RELU
	  miopenCreateOpActivationForward(fusePlanDesc2, &activOp2, miopenActivationRELU);

	  // compile fusion plan
	  auto status = miopenCompileFusionPlan(mio::instance()->handle(), fusePlanDesc2);
	  if (status != miopenStatusSuccess) { return -1; }
	  float alpha = static_cast<float>(1), beta = static_cast<float>(0);
	  float activ_alpha = static_cast<float>(1), activ_beta = static_cast<float>(0), activ_gamma = static_cast<float>(1);

	  // Set the Args
	  miopenSetOpArgsConvForward(fusionArgs2, convoOp2, &alpha, &beta, weights2.data);
	  miopenSetOpArgsActivForward(fusionArgs2, activOp2, &alpha, &beta, activ_alpha, activ_beta, activ_gamma);
	  miopenSetOpArgsBiasForward(fusionArgs2, biasOp2, &alpha, &beta, bias2.data);
	  weights2.uniform();
  }

  // possibly in a loop but with new values for the tensors to be meaningful
  // Here we use the same values to keep the code simple
  for (auto idx = 0; idx < 1000; idx++) {
	//input1.uniform();
	//weights1.uniform();
	//weights2.uniform();
	if(auto ret = miopenExecuteFusionPlan(mio::instance()->handle(), fusePlanDesc1, input1.desc, input1.data, output1.desc, output1.data, fusionArgs1); ret != miopenStatusSuccess) {
		printf("Error while executing first fusion %d\n", ret);
	}
    if(auto ret = miopenExecuteFusionPlan(mio::instance()->handle(), fusePlanDesc2, output1.desc, output1.data, output2.desc, output2.data, fusionArgs2); ret != miopenStatusSuccess) {
    	printf("Error while executing second fusion %d\n", ret);
    }
    clFinish(mio::instance()->GetStream());
    //output2.toHost();
  }

  // Cleanup
  miopenDestroyFusionPlan(fusePlanDesc1);
  miopenDestroyConvolutionDescriptor(conv_desc1);
  miopenDestroyFusionPlan(fusePlanDesc2);
  miopenDestroyConvolutionDescriptor(conv_desc2);
}
