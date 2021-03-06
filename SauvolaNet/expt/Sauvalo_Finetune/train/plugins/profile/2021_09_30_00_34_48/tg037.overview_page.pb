?	!x|{?6@@!x|{?6@@!!x|{?6@@      ??!       "q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0!x|{?6@@??G?3???1?T?#??;@Am7?7M?}?I? l@??@r0*	x?&1H?@2f
/Iterator::Root::Prefetch::FlatMap[0]::Generator??w?G?@!=*?Z?X@)??w?G?@1=*?Z?X@:Preprocessing2O
Iterator::Root::Prefetchs??????!ӎfg????)s??????1ӎfg????:Preprocessing2E
Iterator::Root?NGɫ??!;CJPD???)* ??q??1??-9????:Preprocessing2X
!Iterator::Root::Prefetch::FlatMap???|?B@!?־??X@)???3.l?16ز?~??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"?12.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIX???Z?+@QuM.?4?U@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??G?3?????G?3???!??G?3???      ??!       "	?T?#??;@?T?#??;@!?T?#??;@*      ??!       2	m7?7M?}?m7?7M?}?!m7?7M?}?:	? l@??@? l@??@!? l@??@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qX???Z?+@yuM.?4?U@?
"?
]gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv5/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterH|?y?'??!H|?y?'??0"?
]gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?x??U??!???C?>??0"?
`gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv_att/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?ymi9J??!J?I??c??0"?
]gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???6t ??!?w??5??0"?
]gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?_??o??!?[i4?ó?0"?
\gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv5/Conv2D/Conv2DBackpropInputConv2DBackpropInput?Y[Sݫ??!???e??0"X
:Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv5/Conv2DConv2D?u??э??!˵? +??0"?
\gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv4/Conv2D/Conv2DBackpropInputConv2DBackpropInput-?F9?z?!?b??"??0"-
Adam/gradients/AddNAddN͠?<?!?+????"?
_gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv_att/Conv2D/Conv2DBackpropInputConv2DBackpropInput??g-0x~?!۪????0Q      Y@Y?8??8?@a;?c;?cX@qc??;??yÌh?;+|?"?

device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?12.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Pascal)(: B 