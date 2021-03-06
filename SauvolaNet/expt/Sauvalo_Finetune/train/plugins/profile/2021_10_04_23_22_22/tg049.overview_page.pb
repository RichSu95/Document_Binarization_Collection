?	?aN?&=@?aN?&=@!?aN?&=@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?aN?&=@?cyW=`??1ƨk?}?4@I ?g??? @r0*	6^?IL??@2f
/Iterator::Root::Prefetch::FlatMap[0]::Generator鷯'@!??|Zc?X@)鷯'@1??|Zc?X@:Preprocessing2E
Iterator::Root(????!???? ??)T?{F"4??1??r<?E??:Preprocessing2O
Iterator::Root::Prefetch?s????!??:???)?s????1??:???:Preprocessing2X
!Iterator::Root::Prefetch::FlatMap???Ա*@!,?|?X@) C?*qm?1,X?d??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?28.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?A?={=@Q???p:?Q@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?cyW=`???cyW=`??!?cyW=`??      ??!       "	ƨk?}?4@ƨk?}?4@!ƨk?}?4@*      ??!       2      ??!       :	 ?g??? @ ?g??? @! ?g??? @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?A?={=@y???p:?Q@?
"?
]gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv6/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??V?Ws??!??V?Ws??0"?
]gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv5/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?e?%[??!??ݰ>g??0"?
`gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv_att/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter>?N?h??!V?Db??0"?
]gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter'?H?`??! ??ˋY??0"?
]gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???????!I???????0"X
:Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv6/Conv2DConv2D8M{}I???!?ZI????0"?
\gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv6/Conv2D/Conv2DBackpropInputConv2DBackpropInput??"?l??!??ͪe??0"?
]gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv2/Conv2D/Conv2DBackpropFilterConv2DBackpropFiltertQߎX??!?7??????0"X
:Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv5/Conv2DConv2D$V?\??!Y]?????0"?
_gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv_att/Conv2D/Conv2DBackpropInputConv2DBackpropInput???ě???!r???YC??0Q      Y@YR??3?M@a?)`>?]X@q????KU??y??S?j??"?

device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?28.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Pascal)(: B 