?	?ɨ2??7@?ɨ2??7@!?ɨ2??7@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?ɨ2??7@T?qs*??1G?ŧ ?/@I0G????@r0*	???Ԙʵ@2f
/Iterator::Root::Prefetch::FlatMap[0]::GeneratorV??6o@!?EO?³X@)V??6o@1?EO?³X@:Preprocessing2E
Iterator::Root??V'g??!f?7?`??)>???6??1݊	?ϥ??:Preprocessing2O
Iterator::Root::Prefetch??.?.??!ޣ[R?6??)??.?.??1ޣ[R?6??:Preprocessing2X
!Iterator::Root::Prefetch::FlatMap?(?@!F?#?}?X@)?kC?8c?1??]??ץ?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?33.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIҚ??@@Q?????P@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	T?qs*??T?qs*??!T?qs*??      ??!       "	G?ŧ ?/@G?ŧ ?/@!G?ŧ ?/@*      ??!       2      ??!       :	0G????@0G????@!0G????@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qҚ??@@y?????P@?
"?
]gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv5/Conv2D/Conv2DBackpropFilterConv2DBackpropFilteru}?d?c??!u}?d?c??0"?
]gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?W?T????!?j?\_???0"?
`gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv_att/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterǾ?[??!?*?uش?0"?
]gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?Q9o???!}????9??0"?
]gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???7x??!e???ɗ??0"X
:Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv5/Conv2DConv2D??????!;???N;??0"?
]gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?5O??!?d?????0"?
\gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv5/Conv2D/Conv2DBackpropInputConv2DBackpropInput????aŉ?!rq0?ix??0"X
:Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv4/Conv2DConv2D????/???!?0??????0"
`Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/relu1/Relu-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose????y??!??z?j??Q      Y@Yi%1?1?@a?v?p?HX@qD\)?$??y{Z?????"?

device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?33.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Pascal)(: B 