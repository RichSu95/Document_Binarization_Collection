
?
?void wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)G?*2?8??@??H??Xb]gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv5/Conv2D/Conv2DBackpropFilterhu  /B
?
?void wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)G?*2?8??@??H??Xb]gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv4/Conv2D/Conv2DBackpropFilterhu  /B
?
?void wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)G?*2	?8ѳ@ѳHѳXb`gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv_att/Conv2D/Conv2DBackpropFilterhu  /B
?
?void wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)G?*2?8??@??H??Xb]gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv3/Conv2D/Conv2DBackpropFilterhu  /B
?
?void wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)G?*2?8̣@̣ḤXb]gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv2/Conv2D/Conv2DBackpropFilterhu  /B
?
Dmaxwell_scudnn_winograd_128x128_ldg1_ldg4_mobile_relu_tile148t_nt_v0???*?2
8??@??H??Xb:Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv5/Conv2Dhu  ?A
?
?void wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)G?*2?8??@??H??Xb]gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv1/Conv2D/Conv2DBackpropFilterhu  /B
?
Dmaxwell_scudnn_winograd_128x128_ldg1_ldg4_mobile_relu_tile148t_nt_v0???*?2
8??@??H??Xb\gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv5/Conv2D/Conv2DBackpropInputhu  ?A
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 1024, 2, 1024, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?@*?2?8Ȋ@ȊHȊb`Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/relu1/Relu-0-1-TransposeNCHWToNHWC-LayoutOptimizerhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 1024, 2, 1024, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?@*?2?8??@??H??bgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/bnorm1/FusedBatchNormGradV3-0-1-TransposeNCHWToNHWC-LayoutOptimizerhu  ?B
?
?void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(?*?28??
@??
H??
bWgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/bnorm5/FusedBatchNormGradV3hu  ?B
?
&maxwell_scudnn_128x32_relu_small_nn_v1??R*@2?8??
@??
H??
Xb:Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv4/Conv2Dhu  ?A
?
Dmaxwell_scudnn_winograd_128x128_ldg1_ldg4_mobile_relu_tile148t_nt_v0???*?2
8??
@??
H??
Xb\gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv4/Conv2D/Conv2DBackpropInputhu  ?A
?
?void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(?*?28??
@??
H??
bWgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/bnorm4/FusedBatchNormGradV3hu  ?B
?
Dmaxwell_scudnn_winograd_128x128_ldg1_ldg4_mobile_relu_tile148t_nt_v0???*?2&8??	@??	H??	Xb=Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv_att/Conv2Dhu  ?A
?
&maxwell_scudnn_128x32_relu_small_nn_v1??R*@2?8Ǹ	@Ǹ	HǸ	Xb:Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv3/Conv2Dhu  ?A
?
?void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(?*?28??	@??	H??	bWgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/bnorm3/FusedBatchNormGradV3hu  ?B
?
Dmaxwell_scudnn_winograd_128x128_ldg1_ldg4_mobile_relu_tile148t_nt_v0???*?2&8Ƈ	@Ƈ	HƇ	Xb_gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv_att/Conv2D/Conv2DBackpropInputhu  ?A
?
?void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(?*?28??@??H??bWgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/bnorm2/FusedBatchNormGradV3hu  ?B
?
Dmaxwell_scudnn_winograd_128x128_ldg1_ldg4_mobile_relu_tile148t_nt_v0???*?2
8??@??H??Xb\gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv3/Conv2D/Conv2DBackpropInputhu  ?A
?
Dmaxwell_scudnn_winograd_128x128_ldg1_ldg4_mobile_relu_tile148t_nt_v0???*?2
8ƭ@ƭHƭXb\gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv2/Conv2D/Conv2DBackpropInputhu  ?A
?
?void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(?*?28ŋ@ŋHŋbWgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/bnorm1/FusedBatchNormGradV3hu  ?B
?
Xvoid tensorflow::S2B<float, 2, false>(int, float*, tensorflow::S2BParameters<2>, float*)*?28??@??H??bWgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv5/Conv2D/SpaceToBatchNDhu  ?B
?
&maxwell_scudnn_128x32_relu_small_nn_v1??R*@2?8??@??H??Xb:Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv2/Conv2Dhu  ?A
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 1024, 2, 1024, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?@*?2?8??@??H??bUgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/time1/transpose/transposehu  ?B
?
Wvoid tensorflow::S2B<float, 2, true>(int, float*, tensorflow::S2BParameters<2>, float*)*?28??@??H??bISauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv5/Conv2D/BatchToSpaceNDhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 1024, 2, 1024, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?@*?2?8??@??H??bfSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv_att/BiasAdd-0-0-TransposeNCHWToNHWC-LayoutOptimizerhu  ?B
?
?void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)(?*?28??@??H??bWgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/bnorm0/FusedBatchNormGradV3hu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 1024, 2, 1024, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?@*?2?8Ž@ŽHŽb`Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/relu0/Relu-0-1-TransposeNCHWToNHWC-LayoutOptimizerhu  ?B
?
Dmaxwell_scudnn_winograd_128x128_ldg1_ldg4_mobile_relu_tile148t_nt_v0???*?2
8??@??H??Xb:Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv1/Conv2Dhu  ?A
?
Xvoid tensorflow::S2B<float, 2, false>(int, float*, tensorflow::S2BParameters<2>, float*)*?28??@??H??bISauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv5/Conv2D/SpaceToBatchNDhu  ?B
?
Xvoid tensorflow::S2B<float, 2, false>(int, float*, tensorflow::S2BParameters<2>, float*)*?28??@??H??bWgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv4/Conv2D/SpaceToBatchNDhu  ?B
?
Dmaxwell_scudnn_winograd_128x128_ldg1_ldg4_mobile_relu_tile148t_nt_v0???*?2
8??@??H??Xb\gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv1/Conv2D/Conv2DBackpropInputhu  ?A
?
Wvoid tensorflow::S2B<float, 2, true>(int, float*, tensorflow::S2BParameters<2>, float*)*?28??@??H??bWgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv5/Conv2D/BatchToSpaceNDhu  ?B
?
Wvoid tensorflow::S2B<float, 2, true>(int, float*, tensorflow::S2BParameters<2>, float*)*?28??@??H??bISauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv4/Conv2D/BatchToSpaceNDhu  ?B
?
?void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, 512, true, 1>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(?*?28Ŀ@ĿHĿbESauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/bnorm5/FusedBatchNormV3hu  ?B
?
"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?2?8??@??H??b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_88hu  ?B
?
"Mul_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?2?8ŗ@ŗHŗb;Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/mul_1hu  ?B
?
Xvoid tensorflow::S2B<float, 2, false>(int, float*, tensorflow::S2BParameters<2>, float*)*?28??@??H??bISauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv4/Conv2D/SpaceToBatchNDhu  ?B
?
"Mul_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?2?8ō@ōHōbMgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/mul_1/Mulhu  ?B
?
"Mul_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?2?8??@??H??bRgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/truediv_22/mulhu  ?B
?
Xvoid tensorflow::S2B<float, 2, false>(int, float*, tensorflow::S2BParameters<2>, float*)*?28??@??H??bWgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv3/Conv2D/SpaceToBatchNDhu  ?B
?
"Mul_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?2?8??@??H??bKgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/mul/Mulhu  ?B
?
Wvoid tensorflow::S2B<float, 2, true>(int, float*, tensorflow::S2BParameters<2>, float*)*?28??@??H??bWgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv4/Conv2D/BatchToSpaceNDhu  ?B
?
Wvoid tensorflow::S2B<float, 2, true>(int, float*, tensorflow::S2BParameters<2>, float*)*?28??@??H??bISauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv3/Conv2D/BatchToSpaceNDhu  ?B
?
?void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, 512, true, 1>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(?*?28??@??H??bESauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/bnorm4/FusedBatchNormV3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?288??@??H??bJgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/relu5/ReluGradhu  ?B
?
Xvoid tensorflow::S2B<float, 2, false>(int, float*, tensorflow::S2BParameters<2>, float*)*?28??@??H??bISauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv3/Conv2D/SpaceToBatchNDhu  ?B
?
?void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, 512, true, 1>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(?*?28??@??H??bESauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/bnorm3/FusedBatchNormV3hu  ?B
?
Xvoid tensorflow::S2B<float, 2, false>(int, float*, tensorflow::S2BParameters<2>, float*)*?28??@??H??bWgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv2/Conv2D/SpaceToBatchNDhu  ?B
?
Wvoid tensorflow::S2B<float, 2, true>(int, float*, tensorflow::S2BParameters<2>, float*)*?28??@??H??bWgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv3/Conv2D/BatchToSpaceNDhu  ?B
?
Wvoid tensorflow::S2B<float, 2, true>(int, float*, tensorflow::S2BParameters<2>, float*)*?28??@??H??bISauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv2/Conv2D/BatchToSpaceNDhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?288??@??H??bJgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/relu4/ReluGradhu  ?B
?
?void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, 512, true, 1>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(?*?28??@??H??bESauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/bnorm2/FusedBatchNormV3hu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?!*?2?8??@??H??bbSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv5/Conv2D-0-0-TransposeNCHWToNHWC-LayoutOptimizerhu  ?B
?
?void wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)G?*2?8??@??H??Xb]gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv0/Conv2D/Conv2DBackpropFilterhu  /B
?
"Neg_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?2?8??@??H??bRgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/truediv_22/Neghu  ?B
?
?void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, 512, true, 1>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(?*?28??@??H??bESauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/bnorm1/FusedBatchNormV3hu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?!*?2?8ī@īHībgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/bnorm5/FusedBatchNormGradV3-0-1-TransposeNCHWToNHWC-LayoutOptimizerhu  ?B
?
%Square_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?2?8??@??H??b;Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/pow_1hu  ?B
?
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*?28??@??H??b;Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv5/BiasAddhu  ?B
?
Xvoid tensorflow::S2B<float, 2, false>(int, float*, tensorflow::S2BParameters<2>, float*)*?28??@??H??bISauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv2/Conv2D/SpaceToBatchNDhu  ?B
?
?void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, 512, true, 1>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(?*?28??@??H??bESauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/bnorm0/FusedBatchNormV3hu  ?B
?
Xvoid tensorflow::S2B<float, 2, false>(int, float*, tensorflow::S2BParameters<2>, float*)*?28??@??H??bWgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv1/Conv2D/SpaceToBatchNDhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?288??@??H??bJgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/relu3/ReluGradhu  ?B
?
Wvoid tensorflow::S2B<float, 2, true>(int, float*, tensorflow::S2BParameters<2>, float*)*?28??@??H??bWgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv2/Conv2D/BatchToSpaceNDhu  ?B
?
Wvoid tensorflow::S2B<float, 2, true>(int, float*, tensorflow::S2BParameters<2>, float*)*?28??@??H??bISauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv1/Conv2D/BatchToSpaceNDhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 1024, 2, 1024, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?@*?2?8??@??H??b?gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv2/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizerhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?!*?2?8??@??H??baSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv5/BiasAdd-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?!*?2?8??@??H??bbSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv4/Conv2D-0-0-TransposeNCHWToNHWC-LayoutOptimizerhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 1024, 2, 1024, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?@*?2?8??@??H??bbSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv1/Conv2D-0-0-TransposeNCHWToNHWC-LayoutOptimizerhu  ?B
?
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*?28??@??H??b;Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv4/BiasAddhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?!*?2?8??@??H??b?gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv5/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizerhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?!*?2?8??@??H??b?gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv5/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*?288??@??H??b8Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/relu5/Reluhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?!*?2?8??@??H??b`Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/relu4/Relu-0-1-TransposeNCHWToNHWC-LayoutOptimizerhu  ?B
?
#Sqrt_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?2?8??@??H??b:Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/Sqrthu  ?B
?
?void tensorflow::functor::RowReduceKernel<cub::TransformInputIterator<float, tensorflow::(anonymous namespace)::SubtractAndExpFunctor<float, float>, cub::CountingInputIterator<int, long>, long>, float*, cub::Sum>(cub::TransformInputIterator<float, tensorflow::(anonymous namespace)::SubtractAndExpFunctor<float, float>, cub::CountingInputIterator<int, long>, long>, float*, int, int, cub::Sum, std::iterator_traits<cub::TransformInputIterator<float, tensorflow::(anonymous namespace)::SubtractAndExpFunctor<float, float>, cub::CountingInputIterator<int, long>, long> >::value_type)*?2??8??@??H??b>Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv_att/Softmaxhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?!*?2?8??@??H??b?gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv5/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?!*?2?8??@??H??baSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv4/BiasAdd-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?!*?2?8??@??H??bgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/bnorm4/FusedBatchNormGradV3-0-1-TransposeNCHWToNHWC-LayoutOptimizerhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?288??@??H??bJgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/relu2/ReluGradhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?!*?2?8??@??H??bpgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/relu4/ReluGrad-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?!*?2?8??@??H??b?gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv4/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*?28??@??H??b;Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv3/BiasAddhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?!*?2?8??@??H??bbSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv3/Conv2D-0-0-TransposeNCHWToNHWC-LayoutOptimizerhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*?288??@??H??b8Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/relu4/Reluhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?!*?2?8??@??H??b?gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv4/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizerhu  ?B
?
Xvoid tensorflow::S2B<float, 2, false>(int, float*, tensorflow::S2BParameters<2>, float*)*?28Ö@ÖHÖbISauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv1/Conv2D/SpaceToBatchNDhu  ?B
?
Wvoid tensorflow::S2B<float, 2, true>(int, float*, tensorflow::S2BParameters<2>, float*)*?28@HbWgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv1/Conv2D/BatchToSpaceNDhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?!*?2?8??@??H??b`Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/relu3/Relu-0-1-TransposeNCHWToNHWC-LayoutOptimizerhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?!*?2?8??@??H??bpgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/relu3/ReluGrad-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?!*?2?8??@??H??b?gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv4/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?2?8??@??H??b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_89hu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?!*?2?8??@??H??baSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv3/BiasAdd-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
"Div_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?2?8??@??H??b@Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/truediv_22hu  ?B
?
&Maximum_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?2?8??@??H??b?Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/Maximum_1hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?2?8??@??H??b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_44hu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 1024, 1024, 2, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?`*?2?8??@??H??baSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv1/BiasAdd-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?!*?2?8??@??H??bgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/bnorm3/FusedBatchNormGradV3-0-1-TransposeNCHWToNHWC-LayoutOptimizerhu  ?B
?
"Div_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?2?8??@??H??bXgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/truediv_22/RealDiv_2hu  ?B
?
?void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Max>(float const*, float*, int, int, cub::Max, std::iterator_traits<float const*>::value_type)*?2??8??@??H??b>Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv_att/Softmaxhu  ?B
?
&Maximum_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?2?8??@??H??b=Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/Maximumhu  ?B
?
"Div_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?2?8??@??H??bXgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/truediv_22/RealDiv_1hu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?!*?2?8??@??H??b?gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv3/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizerhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?!*?2?8??@??H??b?gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv3/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 1024, 1024, 2, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?`*?2?8??@??H??b?gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv2/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?!*?2?8??@??H??bbSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv2/Conv2D-0-0-TransposeNCHWToNHWC-LayoutOptimizerhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 1024, 1024, 2, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?`*?2?8??@??H??bpgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/relu1/ReluGrad-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 1024, 2, 1024, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?@*?2?8??@??H??b?gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv1/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizerhu  ?B
?
?void explicit_convolve_sgemm<float, int, 1024, 5, 5, 3, 3, 3, 0, false>(int, int, int, float const*, int, float const*, int, float*, kernel_conv_params, unsigned long long, int, unsigned long long, int, float, float, int, float const*, float const*)>?*2?8??@??H??Xb:Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv0/Conv2Dhu  HB
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 1024, 1024, 2, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?`*?2?8??@??H??b?gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv1/Conv2D/Conv2DBackpropInput-2-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?288??@??H??bJgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/relu1/ReluGradhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?!*?2?8??@??H??b`Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/relu2/Relu-0-1-TransposeNCHWToNHWC-LayoutOptimizerhu  ?B
?
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*?28??@??H??b;Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv2/BiasAddhu  ?B
?
"Mul_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?2?8??@??H??b9Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/mulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*?288??@??H??b8Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/relu3/Reluhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?!*?2?8??@??H??bgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/bnorm2/FusedBatchNormGradV3-0-1-TransposeNCHWToNHWC-LayoutOptimizerhu  ?B
?
"Mul_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?2?8??@??H??bMgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/mul/Mul_1hu  ?B
?
?void tensorflow::functor::RowReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*?2??8??@??H??bHgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv_att/Sumhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?!*?2?8??@??H??bpgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/relu2/ReluGrad-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?!*?2?8??@??H??b?gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv3/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?!*?2?8??@??H??baSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv2/BiasAdd-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?!*?2?8??@??H??b?gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv2/Conv2D/Conv2DBackpropInput-2-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*?288??@??H??b8Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/relu2/Reluhu  ?B
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, cub::Sum>::Policy600, float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float>(float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float)0*?28??@??H??bUgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv5/BiasAdd/BiasAddGradhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 1024, 1024, 2, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?`*?2?8??@??H??b?gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv1/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)?*?268@HbUgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv4/BiasAdd/BiasAddGradhu  ?B
?
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*?28??@??H??b;Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv1/BiasAddhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 1024, 1024, 2, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?`*?2?8??@??H??b?gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv_att/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
"Cast_GPU_DT_DOUBLE_DT_FLOAT_kernel*?2?8??@??H??b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/Cast_3hu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 1024, 1024, 2, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?`*?2?8??@??H??b=Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/time1/transposehu  ?B
?
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?8??@??H??b;Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/attention/mulhu  ?B
?
"Cast_GPU_DT_FLOAT_DT_DOUBLE_kernel
*?2?8??@??H??bOgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/Cast_3/Casthu  ?B
?
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?8??@??H??bHgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv_att/mulhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 1024, 1024, 2, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?`*?2?8??@??H??bpgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/relu0/ReluGrad-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?8@HbJgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv_att/mul_1hu  ?B
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<double, double, int, tensorflow::functor::Sum<double> >::Policy600, double*, double*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<double>, double>(double*, double*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<double>, double)P*?28??@??H??bKgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/mul/Sumhu  ?B
?
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?8??@??H??bOgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/attention/mul/Mul_1hu  ?B
?
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?8??@??H??bMgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/attention/mul/Mulhu  ?B
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<double, double, int, tensorflow::functor::Sum<double> >::Policy600, double*, double*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<double>, double>(double*, double*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<double>, double)P*?28??@??H??bTgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/truediv_22/Sum_1hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?288??@??H??bJgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/relu0/ReluGradhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*?288??@??H??b8Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/relu1/Reluhu  ?B
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, cub::Sum>::Policy600, float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float>(float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float)0*?28??@??H??bUgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv3/BiasAdd/BiasAddGradhu  ?B
?
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*?28??@??H??b;Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv0/BiasAddhu  ?B
?
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*?28??@??H??b>Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv_att/BiasAddhu  ?B
?
?void tensorflow::(anonymous namespace)::GenerateNormalizedProb<float, float, 4>(float const*, float const*, float const*, float*, int, int, bool)*?2?8??@??H??b>Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv_att/Softmaxhu  ?B
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, cub::Sum>::Policy600, float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float>(float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float)0*?28??@??H??bUgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv2/BiasAdd/BiasAddGradhu  ?B
?
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel
*?2?8??@??H??bHgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv_att/subhu  ?B
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, cub::Sum>::Policy600, float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float>(float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float)0*?28??@??H??bUgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv1/BiasAdd/BiasAddGradhu  ?B
?
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)?*?2!8??@??H??bXgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv_att/BiasAdd/BiasAddGradhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*?288??@??H??b8Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/relu0/Reluhu  ?B
?
?void tensorflow::functor::scan_kernel<double, tensorflow::functor::Sum<double>, 64, 4>(double const*, double*, int, int, int, bool, bool, tensorflow::functor::Sum<double>)(?*@2?8??@??H??b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/Cumsumhu  ?B
?
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)?*?2!8??@??H??bUgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv0/BiasAdd/BiasAddGradhu  ?B
?
?void cudnn::cnn::im2col4d_kernel<float, long>(cudnn::cnn::im2col4d_params, cudnnConvolutionStruct, cudnnTensorStruct, float const*, float*)%*?2?8??@??H??Xb:Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv0/Conv2Dhu  ?B
?
?void tensorflow::functor::scan_kernel<double, tensorflow::functor::Sum<double>, 64, 4>(double const*, double*, int, int, int, bool, bool, tensorflow::functor::Sum<double>)(?*@2?8??@??H??b>Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/Cumsum_2hu  ?B
?
?void tensorflow::functor::scan_kernel<double, tensorflow::functor::Sum<double>, 64, 4>(double const*, double*, int, int, int, bool, bool, tensorflow::functor::Sum<double>)(?*@2?8??@??H??b?Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/Cumsum_26hu  ?B
?
?void tensorflow::functor::scan_kernel<double, tensorflow::functor::Sum<double>, 64, 4>(double const*, double*, int, int, int, bool, bool, tensorflow::functor::Sum<double>)(?*@2?8??@??H??b?Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/Cumsum_24hu  ?B
?
?void tensorflow::functor::scan_kernel<double, tensorflow::functor::Sum<double>, 64, 4>(double const*, double*, int, int, int, bool, bool, tensorflow::functor::Sum<double>)(?*@2?8??@??H??b?Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/Cumsum_27hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?q@?qH?qbGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_107hu  ?B
?
?void tensorflow::functor::scan_kernel<double, tensorflow::functor::Sum<double>, 64, 4>(double const*, double*, int, int, int, bool, bool, tensorflow::functor::Sum<double>)(?*@2?8?q@?qH?qb>Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/Cumsum_3hu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>) *?2?8?q@?qH?qb;Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/attention/Sumhu  ?B
?
?void tensorflow::functor::scan_kernel<double, tensorflow::functor::Sum<double>, 64, 4>(double const*, double*, int, int, int, bool, bool, tensorflow::functor::Sum<double>)(?*@2?8?p@?pH?pb>Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/Cumsum_1hu  ?B
?
?void tensorflow::functor::scan_kernel<double, tensorflow::functor::Sum<double>, 64, 4>(double const*, double*, int, int, int, bool, bool, tensorflow::functor::Sum<double>)(?*@2?8?h@?hH?hb?Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/Cumsum_25hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?\@?\H?\bQgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/attention/BroadcastTohu  ?B
?
"Div_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?2X8?L@?LH?Lb?Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/truediv_1hu  ?B
?
"Div_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?2X8?J@?JH?Jb@Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/truediv_19hu  ?B
?
"Div_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?2X8?I@?IH?Ib?Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/truediv_3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 4ul> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 4ul> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *?288?G@?GH?Gb;Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/Pad_1hu  ?B
?
"Div_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?2X8?G@?GH?Gb@Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/truediv_13hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?G@?GH?Gb;Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_7hu  ?B
?
"Div_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?2X8?F@?FH?Fb?Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/truediv_9hu  ?B
?
"Div_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?2X8?F@?FH?Fb@Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/truediv_14hu  ?B
?
"Div_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?2X8?F@?FH?Fb?Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/truediv_6hu  ?B
?
"Div_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?2X8?E@?EH?Eb@Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/truediv_20hu  ?B
?
"Div_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?2X8?E@?EH?Eb?Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/truediv_5hu  ?B
?
"Div_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?2X8?E@?EH?Eb@Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/truediv_12hu  ?B
?
"Div_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?2X8?E@?EH?Eb@Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/truediv_17hu  ?B
?
"Div_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?2X8?E@?EH?Eb@Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/truediv_18hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?E@?EH?Eb;Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_2hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?E@?EH?Eb<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_27hu  ?B
~
"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?E@?EH?Eb;Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_4hu  ?B
?
"Div_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?2X8?E@?EH?Eb@Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/truediv_16hu  ?B
?
"Div_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?2X8?E@?EH?Eb?Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/truediv_7hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 4ul> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 4ul> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *?288?E@?EH?Eb<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/Pad_13hu  ?B
?
"Div_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?2X8?D@?DH?Db=Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/truedivhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 3, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 3, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 3, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 3, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *?288?D@?DH?Db(gradient_tape/hinge/weighted_loss/Tile_1hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?D@?DH?Db<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_13hu  ?B
?
"Div_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?2X8?C@?CH?Cb?Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/truediv_8hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?C@?CH?Cb<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_41hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?C@?CH?Cb<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_47hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?C@?CH?Cb;Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_6hu  ?B
?
"Div_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?2X8?C@?CH?Cb@Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/truediv_11hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?C@?CH?Cb<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_15hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?B@?BH?Bb<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_62hu  ?B
?
"Div_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?2X8?B@?BH?Bb@Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/truediv_21hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?B@?BH?Bb<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_43hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?B@?BH?Bb<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_43hu  ?B
?
"Div_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?2X8?B@?BH?Bb?Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/truediv_2hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?B@?BH?Bb;Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_3hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?B@?BH?Bb<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_31hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?B@?BH?Bb<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_60hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?B@?BH?Bb<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_81hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?A@?AH?Ab<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_21hu  ?B
?
"Div_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?2X8?A@?AH?Ab@Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/truediv_15hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?A@?AH?Ab<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_19hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?A@?AH?Ab<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_14hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?A@?AH?Ab<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_20hu  ?B
~
"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?A@?AH?Ab;Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_3hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?A@?AH?Ab<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_51hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?A@?AH?Ab<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_72hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?A@?AH?Ab<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_25hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?A@?AH?Ab<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_34hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?A@?AH?Ab<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_46hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?A@?AH?Ab<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_79hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?A@?AH?Ab<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_70hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?A@?AH?Ab<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_85hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?A@?AH?Ab;Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_1hu  ?B
?
"Div_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?2X8?A@?AH?Ab?Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/truediv_4hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?A@?AH?Ab<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_12hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?A@?AH?Ab<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_21hu  ?B
~
"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?A@?AH?Ab;Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_5hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?@@?@H?@b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_71hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?@@?@H?@b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_73hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?@@?@H?@b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_18hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?@@?@H?@b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_42hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?@@?@H?@b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_12hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?@@?@H?@b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_22hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?@@?@H?@b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_32hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?@@?@H?@b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_48hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?@@?@H?@b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_67hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?@@?@H?@b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_35hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?@@?@H?@b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_24hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?@@?@H?@b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_59hu  ?B
|
"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?@@?@H?@b9Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/subhu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?@@?@H?@b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_27hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?@@?@H?@b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_49hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?@@?@H?@b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_57hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?@@?@H?@b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_10hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?@@?@H?@b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_84hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?@@?@H?@b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_11hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?@@?@H?@b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_28hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?@@?@H?@b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_32hu  ?B
?
"Div_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?2X8?@@?@H?@b@Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/truediv_10hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?@@?@H?@b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_33hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?@@?@H?@b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_50hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?@@?@H?@b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_68hu  ?B
~
"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?@@?@H?@b;Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_7hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28??@??H??b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_36hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28??@??H??b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_64hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28??@??H??b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_10hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28??@??H??b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_41hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28??@??H??b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_66hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28??@??H??b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_58hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28??@??H??b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_24hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28??@??H??b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_25hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28??@??H??b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_63hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28??@??H??b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_74hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28??@??H??b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_14hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28??@??H??b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_36hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28??@??H??b;Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_8hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28??@??H??b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_18hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28??@??H??b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_15hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28??@??H??b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_26hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28??@??H??b;Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_4hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28??@??H??b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_23hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28??@??H??b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_69hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28??@??H??b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_83hu  ?B
~
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28??@??H??b9Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/addhu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28??@??H??b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_26hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28??@??H??b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_30hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28??@??H??b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_75hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28??@??H??b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_80hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?>@?>H?>b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_19hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?>@?>H?>b;Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_9hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?>@?>H?>b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_37hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?>@?>H?>b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_53hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?>@?>H?>b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_87hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?>@?>H?>b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_23hu  ?B
~
"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?>@?>H?>b;Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_2hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?>@?>H?>b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_34hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?>@?>H?>b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_38hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?>@?>H?>b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_13hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?>@?>H?>b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_31hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?>@?>H?>b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_52hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?>@?>H?>b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_29hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?>@?>H?>b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_37hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?>@?>H?>b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_39hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?>@?>H?>b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_65hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?>@?>H?>b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_39hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?>@?>H?>b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_54hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?>@?>H?>b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_61hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?>@?>H?>b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_77hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?>@?>H?>b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_82hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?>@?>H?>b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_30hu  ?B
~
"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?>@?>H?>b;Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_1hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?>@?>H?>b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_29hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?>@?>H?>b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_42hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?>@?>H?>b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_78hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?>@?>H?>b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_16hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?>@?>H?>b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_17hu  ?B
~
"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?>@?>H?>b;Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_8hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?=@?=H?=b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_55hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?=@?=H?=b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_17hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?=@?=H?=b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_33hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?=@?=H?=b;Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_5hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?=@?=H?=b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_28hu  ?B
~
"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?=@?=H?=b;Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_9hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?=@?=H?=b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_40hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?=@?=H?=b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_44hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?=@?=H?=b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_11hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?=@?=H?=b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_22hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?=@?=H?=b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_56hu  ?B
~
"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?=@?=H?=b;Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_6hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?=@?=H?=b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_16hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?=@?=H?=b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_45hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?=@?=H?=b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_20hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?=@?=H?=b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_35hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?=@?=H?=b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_38hu  ?B
?
$AddV2_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?<@?<H?<b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/add_40hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?<@?<H?<b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_76hu  ?B

"Sub_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?<@?<H?<b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/sub_86hu  ?B
?
~void cudnn::winograd::generateWinogradTilesKernel<1, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)(?D* 28?;@?;H?;Xb:Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv5/Conv2Dhu ??B
?
~void cudnn::winograd::generateWinogradTilesKernel<1, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)(?D* 28?;@?;H?;Xb\gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv5/Conv2D/Conv2DBackpropInputhu ??B
?
~void cudnn::winograd::generateWinogradTilesKernel<1, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)(?D* 28?:@?:H?:Xb\gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv3/Conv2D/Conv2DBackpropInputhu ??B
?
~void cudnn::winograd::generateWinogradTilesKernel<1, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)(?D* 28?:@?:H?:Xb\gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv4/Conv2D/Conv2DBackpropInputhu ??B
?
~void cudnn::winograd::generateWinogradTilesKernel<1, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)(?D* 28?:@?:H?:Xb:Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv1/Conv2Dhu ??B
?
~void cudnn::winograd::generateWinogradTilesKernel<1, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)(?D* 28?:@?:H?:Xb_gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv_att/Conv2D/Conv2DBackpropInputhu ??B
?
~void cudnn::winograd::generateWinogradTilesKernel<1, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)(?D* 28?:@?:H?:Xb\gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv1/Conv2D/Conv2DBackpropInputhu ??B
?
~void cudnn::winograd::generateWinogradTilesKernel<1, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)(?D* 28?:@?:H?:Xb\gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv2/Conv2D/Conv2DBackpropInputhu ??B
?
~void cudnn::winograd::generateWinogradTilesKernel<1, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)(?D* 28?9@?9H?9Xb=Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv_att/Conv2Dhu ??B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 4ul> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 4ul> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *?288?9@?9H?9b9Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/Padhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?8@?8H?8bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_196hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?5@?5H?5bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_25hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 4ul> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 4ul> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *?288?5@?5H?5b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/Pad_12hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?3@?3H?3bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_111hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?2@?2H?2bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_178hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?2@?2H?2bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_23hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?1@?1H?1bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_94hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?1@?1H?1bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_152hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?1@?1H?1bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_64hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?0@?0H?0bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_106hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?0@?0H?0bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_179hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?0@?0H?0bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_167hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?/@?/H?/bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_30hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?/@?/H?/bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_47hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?/@?/H?/bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_88hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?/@?/H?/bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_40hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?/@?/H?/bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_120hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?/@?/H?/bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_110hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?/@?/H?/bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_17hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?.@?.H?.bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_71hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?.@?.H?.bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_151hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?.@?.H?.bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_190hu  ?B
?
?void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *?2 8?.@?.H?.b	hinge/Allhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?.@?.H?.bESauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_4hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?.@?.H?.bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_78hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?-@?-H?-bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_28hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?-@?-H?-bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_31hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?-@?-H?-bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_149hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?-@?-H?-bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_112hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?-@?-H?-bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_176hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?-@?-H?-bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_57hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?,@?,H?,bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_46hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?,@?,H?,bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_150hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?,@?,H?,bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_56hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?,@?,H?,bESauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_2hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?,@?,H?,bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_38hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?+@?+H?+bESauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?+@?+H?+bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_118hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?+@?+H?+bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_140hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?*@?*H?*bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_109hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?)@?)H?)bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_105hu  ?B
?
?void tensorflow::(anonymous namespace)::DynamicStitchKernel<int>(int, int, tensorflow::GpuDeviceArrayStruct<int, 8>, tensorflow::GpuDeviceArrayStruct<int const*, 8>, int*)*?28?)@?)H?)bSgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/attention/DynamicStitchhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28?)@?)H?)b%Adam/Adam/update_20/ResourceApplyAdamhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28?(@?(H?(b%Adam/Adam/update_16/ResourceApplyAdamhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28?'@?'H?'b%Adam/Adam/update_12/ResourceApplyAdamhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?&@?&H?&bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_191hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?&@?&H?&bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_164hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?&@?&H?&bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_161hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?&@?&H?&bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_197hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?&@?&H?&bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_68hu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28?%@?%H?%b$Adam/Adam/update_8/ResourceApplyAdamhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?%@?%H?%bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_80hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?%@?%H?%bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_95hu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28?%@?%H?%b$Adam/Adam/update_4/ResourceApplyAdamhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?$@?$H?$bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_69hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?$@?$H?$bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_177hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?$@?$H?$bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_51hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?$@?$H?$bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_73hu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28?$@?$H?$b%Adam/Adam/update_25/ResourceApplyAdamhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?$@?$H?$bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_75hu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28?$@?$H?$b%Adam/Adam/update_24/ResourceApplyAdamhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?$@?$H?$bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_24hu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28?#@?#H?#b%Adam/Adam/update_28/ResourceApplyAdamhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?#@?#H?#bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_86hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?#@?#H?#bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_87hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?#@?#H?#bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_97hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?#@?#H?#bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_62hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?#@?#H?#bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_182hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?#@?#H?#bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_37hu  ?B

%Square_GPU_DT_DOUBLE_DT_DOUBLE_kernel*?28?#@?#H?#b9Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/powhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?#@?#H?#bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_42hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?#@?#H?#bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_79hu  ?B
?
?void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*?2 8?#@?#H?#bSum_3hu  ?B
?
?void tensorflow::(anonymous namespace)::DynamicStitchKernel<int>(int, int, tensorflow::GpuDeviceArrayStruct<int, 8>, tensorflow::GpuDeviceArrayStruct<int const*, 8>, int*)*?28?#@?#H?#b!gradient_tape/hinge/DynamicStitchhu  ?B
\
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*?2X8?#@?#H?#bgradient_tape/hinge/truedivhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?#@?#H?#bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_173hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?#@?#H?#bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_185hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?#@?#H?#bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_65hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?#@?#H?#bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_89hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?#@?#H?#bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_96hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?"@?"H?"bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_175hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?"@?"H?"bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_192hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?"@?"H?"bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_49hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?"@?"H?"bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_77hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?"@?"H?"bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_104hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?"@?"H?"bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_156hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?"@?"H?"bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_158hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?"@?"H?"bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_21hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?"@?"H?"bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_43hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?"@?"H?"bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_66hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?"@?"H?"bESauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_1hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?"@?"H?"bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_163hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?"@?"H?"bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_172hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?"@?"H?"bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_174hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?"@?"H?"bESauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_8hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?"@?"H?"bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_83hu  ?B
M
!Equal_GPU_DT_FLOAT_DT_BOOL_kernel*?28?"@?"H?"bhinge/Equalhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?"@?"H?"bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_100hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?"@?"H?"bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_132hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?"@?"H?"bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_145hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?"@?"H?"bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_165hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?"@?"H?"bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_166hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?"@?"H?"bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_186hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?"@?"H?"bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_32hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?"@?"H?"bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_33hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?"@?"H?"bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_58hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?"@?"H?"bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_70hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?"@?"H?"bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_93hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_187hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_84hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_169hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_188hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_193hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_195hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_59hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_92hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_const_op<double>, Eigen::TensorMap<Eigen::Tensor<double, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_const_op<double>, Eigen::TensorMap<Eigen::Tensor<double, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!b?Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/ones_likehu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_76hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_11hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_134hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_141hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_155hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_159hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_181hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_34hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_35hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_53hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_91hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_114hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_130hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_115hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_119hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_122hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_29hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_50hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_74hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_82hu  ?B
?
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?!@?!H?!bCSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/difference_thresh/subhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_10hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_127hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_129hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_139hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_147hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_154hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_160hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_168hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_170hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_183hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_19hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_55hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bESauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_6hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?!@?!H?!bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_67hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288? @? H? bESauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_5hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288? @? H? bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_131hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288? @? H? bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_22hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288? @? H? bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_26hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288? @? H? bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_13hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288? @? H? bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_146hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288? @? H? bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_44hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288? @? H? bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_60hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288? @? H? bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_85hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288? @? H? bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_98hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288? @? H? bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_116hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288? @? H? bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_142hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288? @? H? bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_103hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288? @? H? bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_124hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288? @? H? bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_61hu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28? @? H? b$Adam/Adam/update_2/ResourceApplyAdamhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28? @? H? b%Adam/Adam/update_22/ResourceApplyAdamhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288? @? H? bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_128hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288? @? H? bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_133hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288? @? H? bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_15hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288? @? H? bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_157hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288? @? H? bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_184hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288? @? H? bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_48hu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28? @? H? b$Adam/Adam/update_6/ResourceApplyAdamhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288? @? H? bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_143hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288? @? H? bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_101hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288? @? H? bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_113hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288? @? H? bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_121hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288? @? H? bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_123hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288? @? H? bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_137hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288? @? H? bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_20hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288? @? H? bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_41hu  ?B
O
!Equal_GPU_DT_FLOAT_DT_BOOL_kernel*?28?@?H?bhinge/Equal_1hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?@?H?bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_16hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?@?H?bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_125hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?@?H?bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_194hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?@?H?bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_52hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?@?H?bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_14hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?@?H?bESauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_7hu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28?@?H?b%Adam/Adam/update_10/ResourceApplyAdamhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?@?H?bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_138hu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28?@?H?b%Adam/Adam/update_14/ResourceApplyAdamhu  ?B
~
"Cast_GPU_DT_FLOAT_DT_DOUBLE_kernel
*?2?8?@?H?b:Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/Casthu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?@?H?bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_136hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?@?H?bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_148hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?@?H?bGSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_102hu  ?B
`
.SquaredDifference_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?bSquaredDifferencehu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?@?H?bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_12hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<double const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?@?H?bFSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/strided_slice_39hu  ?B
?
?void tensorflow::functor::ColumnReduceMax16ColumnsKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)?!* 28?@?H?bUgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv1/BiasAdd/BiasAddGradhu ?	B
?
?void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*?2 8?@?H?bSum_8hu  ?B
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?bmul_4hu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?28?@?H?Xb:Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv0/Conv2Dhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_const_op<double>, Eigen::TensorMap<Eigen::Tensor<double, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_const_op<double>, Eigen::TensorMap<Eigen::Tensor<double, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?288?@?H?bBSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/ones_like_11hu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28?@?H?b%Adam/Adam/update_27/ResourceApplyAdamhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?28?@?H?Xb:Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv5/Conv2Dhu  ?B
I
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?bAdam/Powhu  ?B
c
%SelectV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?8?@?H?bgradient_tape/hinge/SelectV2hu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28?@?H?b%Adam/Adam/update_18/ResourceApplyAdamhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?28?@?H?Xb]gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv3/Conv2D/Conv2DBackpropFilterhu  ?B
G
 Less_GPU_DT_FLOAT_DT_BOOL_kernel*?28?@?H?bLess_1hu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?28?@?H?Xb]gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv2/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?28?@?H?Xb]gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv4/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?28?@?H?Xb_gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv_att/Conv2D/Conv2DBackpropInputhu  ?B
T
$LogicalOr_GPU_DT_BOOL_DT_BOOL_kernel*?2X8?@?H?bhinge/LogicalOrhu  ?B
J
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?b	hinge/mulhu  ?B
F
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?bsub_2hu  ?B
?
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?bCSauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/difference_thresh/mulhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28?@?H?b$Adam/Adam/update_5/ResourceApplyAdamhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?28?@?H?Xb:Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv3/Conv2Dhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?28?@?H?Xb:Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv4/Conv2Dhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28?@?H?b%Adam/Adam/update_21/ResourceApplyAdamhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28?@?H?b%Adam/Adam/update_26/ResourceApplyAdamhu  ?B
?
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?bWgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/difference_thresh/mul/Mul_1hu  ?B
i
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*?28?@?H?b Adam/Adam/update_26/GreaterEqualhu  ?B
X
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?bAdam/Adam/update_26/mulhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?28?@?H?Xb=Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv_att/Conv2Dhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?28?@?H?Xb\gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?28?@?H?Xb`gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv_att/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?28?@?H?Xb:Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv1/Conv2Dhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?28?@?H?Xb:Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv2/Conv2Dhu  ?B
H
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*?2?8?@?H?bCast_1hu  ?B
\
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?bgradient_tape/hinge/mul/Mulhu  ?B
K
#Greater_GPU_DT_FLOAT_DT_BOOL_kernel*?28?@?H?bGreaterhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?28?@?H?Xb]gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv1/Conv2D/Conv2DBackpropFilterhu  ?B
i
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*?28?@?H?b gradient_tape/hinge/GreaterEqualhu  ?B
J
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?b	hinge/subhu  ?B
R
$Maximum_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?bhinge/Maximumhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?28?@?H?Xb]gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv5/Conv2D/Conv2DBackpropFilterhu  ?B
?
Uvoid cask_cudnn::computeOffsetsKernel<false, false>(cask_cudnn::ComputeOffsetsParams)*?2X8?@?H?Xb:Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv4/Conv2Dhu  ?B
E
 Less_GPU_DT_FLOAT_DT_BOOL_kernel*?28?@?H?bLesshu  ?B
G
!Equal_GPU_DT_FLOAT_DT_BOOL_kernel*?28?@?H?bEqualhu  ?B
i
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*?28?@?H?b Adam/Adam/update_28/GreaterEqualhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28?@?H?b%Adam/Adam/update_11/ResourceApplyAdamhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28?@?H?b%Adam/Adam/update_15/ResourceApplyAdamhu  ?B
?
?void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*?2 8?@?H?bMean_1hu  ?B
?
?void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*?2 8?@?H?bhinge/weighted_loss/Sumhu  ?B
D
 Log_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?bLoghu  ?B
?
?void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*?2 8?@?H?bWgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/difference_thresh/mul/Sum_1hu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?28?@?H?Xb\gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv1/Conv2D/Conv2DBackpropInputhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28?@?H?b$Adam/Adam/update_3/ResourceApplyAdamhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28?@?H?b%Adam/Adam/update_19/ResourceApplyAdamhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28?@?H?b%Adam/Adam/update_23/ResourceApplyAdamhu  ?B
?
?void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*?2 8?@?H?bMean_3hu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?28?@?H?Xb]gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv0/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*?2 8?@?H?bSum_2hu  ?B
?
?void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*?2 8?@?H?bSum_9hu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28?@?H?b"Adam/Adam/update/ResourceApplyAdamhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28?@?H?b$Adam/Adam/update_1/ResourceApplyAdamhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28?@?H?b$Adam/Adam/update_7/ResourceApplyAdamhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28?@?H?b$Adam/Adam/update_9/ResourceApplyAdamhu  ?B
?
?void tensorflow::functor::CleanupSegments<bool*, bool*, tensorflow::functor::And>(bool*, bool*, int, int, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type)* 28?@?H?b	hinge/Allhu  HB
?
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?bUgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/difference_thresh/mul/Mulhu  ?B
X
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?bAdam/Adam/update_27/mulhu  ?B
?
Uvoid cask_cudnn::computeOffsetsKernel<false, false>(cask_cudnn::ComputeOffsetsParams)*?2X8?@?H?Xb:Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv3/Conv2Dhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28?@?H?b%Adam/Adam/update_13/ResourceApplyAdamhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28?@?H?b%Adam/Adam/update_17/ResourceApplyAdamhu  ?B
?
?void tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)* 28?@?H?bSum_3hu  HB
?
?void tensorflow::functor::CleanupSegments<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)* 28?@?H?bMean_3hu  HB
?
?void tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)* 28?@?H?bSum_8hu  HB
?
Uvoid cask_cudnn::computeOffsetsKernel<false, false>(cask_cudnn::ComputeOffsetsParams)*?2X8?@?H?Xb:Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv2/Conv2Dhu  ?B
?
?void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)?!* 28?@?H?bUgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv2/BiasAdd/BiasAddGradhu ?	B
H
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*?2?8?@?H?bCast_7hu  ?B
?
?void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)?!* 28?@?H?bUgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv5/BiasAdd/BiasAddGradhu ?	B
?
?void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)?!* 28?@?H?bUgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv3/BiasAdd/BiasAddGradhu ?	B
H
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*?2?8?@?H?bCast_2hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *?288?@?H?bgradient_tape/hinge/zeros_likehu  ?B
\
 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*?2X8?@?H?bgradient_tape/hinge/sub/Neghu  ?B
X
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?bAdam/Adam/update_28/mulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?@?H?b2gradient_tape/hinge/weighted_loss/value/div_no_nanhu  ?B
i
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*?28?@?H?b Adam/Adam/update_27/GreaterEqualhu  ?B
H
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?badd_1hu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?28?@?H?Xb\gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv3/Conv2D/Conv2DBackpropInputhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?28?@?H?Xb\gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv4/Conv2D/Conv2DBackpropInputhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?28?@?H?Xb\gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv5/Conv2D/Conv2DBackpropInputhu  ?B
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?bmul_1hu  ?B
F
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?bsub_3hu  ?B
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?bmul_5hu  ?B
F
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?baddhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?@?H?bAssignAddVariableOp_10hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?@?H?bAdam/Adam/AssignAddVariableOphu  ?B
Y
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*?28?@?H?bAdam/Adam/update_28/Casthu  ?B
H
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*?2?8?@?H?bCast_8hu  ?B
F
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*?28?@?H?bCasthu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?@?H?bdiv_no_nan_3hu  ?B
J
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?b	truediv_6hu  ?B
H
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?btruedivhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?@?H?bdiv_no_nan_2hu  ?B
?
 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*?2X8?@?H?bUgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/difference_thresh/sub/Neghu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?@?H?bAssignAddVariableOp_2hu  ?B
Q
'Reciprocal_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?b	truediv_3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?@?H?bAssignAddVariableOp_7hu  ?B
?
"Cast_GPU_DT_DOUBLE_DT_FLOAT_kernel*?28?@?H?b?gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/Cast_1/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_double_Casthu  ?B
?
"Cast_GPU_DT_DOUBLE_DT_FLOAT_kernel*?28?@?H?b?gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/Cast_2/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_double_Casthu  ?B
K
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?b
Adam/Pow_1hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?@?H?bdiv_no_nan_4hu  ?B
?
?void tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)* 28?@?H?bWgradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/difference_thresh/mul/Sum_1hu  HB
Y
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*?28?@?H?bAdam/Adam/update_26/Casthu  ?B
P
%LogicalAnd_GPU_DT_BOOL_DT_BOOL_kernel*?28?@?H?b
LogicalAndhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?@?H?b
div_no_nanhu  ?B
K
"AddV2_GPU_DT_INT64_DT_INT64_kernel
*?28?@?H?bAdam/addhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?@?H?bdiv_no_nan_1hu  ?B
?
?void tensorflow::functor::CleanupSegments<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)* 28?@?H?bMean_1hu  HB
M
!Cast_GPU_DT_INT64_DT_FLOAT_kernel*?28?@?H?bAdam/Cast_1hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?@?H?bhinge/weighted_loss/valuehu  ?B
J
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?b	truediv_5hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?@?H?bAssignAddVariableOp_8hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?@?H?bAssignAddVariableOp_9hu  ?B
H
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?badd_2hu  ?B
H
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?badd_7hu  ?B
H
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?badd_6hu  ?B
Y
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*?28?@?H?bAdam/Adam/update_27/Casthu  ?B
I
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*?28?@?H?bCast_12hu  ?B
g
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*?28?@?H?b%hinge/weighted_loss/num_elements/Casthu  ?B
J
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?b	truediv_2hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?@?H?bAssignAddVariableOphu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?@?H?bAssignAddVariableOp_5hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?@?H?bAssignAddVariableOp_6hu  ?B
?
?void tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)* 28?@?H?bSum_9hu  HB
?
?void tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)* 28?@?H?bhinge/weighted_loss/Sumhu  HB

"Cast_GPU_DT_FLOAT_DT_DOUBLE_kernel
*?28?@?H?b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/Cast_1hu  ?B
H
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?badd_4hu  ?B
H
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?badd_5hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?@?H?bAssignAddVariableOp_1hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?@?H?bAssignAddVariableOp_3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?@?H?bAssignAddVariableOp_4hu  ?B
?
?void tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)* 28?@?H?bSum_2hu  HB
Q
'Reciprocal_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?b	truediv_4hu  ?B
H
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?badd_3hu  ?B
D
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?bMulhu  ?B

"Cast_GPU_DT_FLOAT_DT_DOUBLE_kernel
*?28?@?H?b<Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/sauvola/Cast_2hu  ?B
J
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?b	truediv_1hu  ?B
Z
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*?28?@?H?bgradient_tape/hinge/Casthu  ?B
H
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*?28?@?H?bCast_3hu  ?B
I
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*?28?@?H?bCast_15hu  ?B
H
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*?28?@?H?bCast_9hu  ?B