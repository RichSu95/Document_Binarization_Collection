?	!?> ??E@!?> ??E@!!?> ??E@	?H7?7????H7?7???!?H7?7???"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0!?> ??E@?e??S9??1??SV???@I?I?5??$@Y???????r0*	?$??Ӽ@2f
/Iterator::Root::Prefetch::FlatMap[0]::Generator???`?4@!m؏Ag?X@)???`?4@1m؏Ag?X@:Preprocessing2E
Iterator::Rootj?q?????!>?,; ??)??)?J=??1+?KbK??:Preprocessing2O
Iterator::Root::Prefetch/ܹ0ҋ??!?
?'j??)/ܹ0ҋ??1?
?'j??:Preprocessing2X
!Iterator::Root::Prefetch::FlatMapm??p9@!?M???X@)?@?p?1????????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?23.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?H7?7???I4?????8@Q????hR@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?e??S9???e??S9??!?e??S9??      ??!       "	??SV???@??SV???@!??SV???@*      ??!       2      ??!       :	?I?5??$@?I?5??$@!?I?5??$@B      ??!       J	??????????????!???????R      ??!       Z	??????????????!???????b      ??!       JGPUY?H7?7???b q4?????8@y????hR@?
"?
]gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv6/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterr???b??!r???b??0"?
]gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv5/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??;??
??! ??\?6??0"?
`gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv_att/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter????SJ??!*??WA???0"?
]gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter-I?%ӌ?!???M߲?0"?
]gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?8uۇ???!?\_I????0"X
:Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv6/Conv2DConv2D??"??Ԅ?!κ#}????0"?
\gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv6/Conv2D/Conv2DBackpropInputConv2DBackpropInput??F? ???!Ŕ???*??0"?
]gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??2?-???!??RG؞??0"X
:Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv5/Conv2DConv2DX :`F??!?4Zӛ???0"?
_gradient_tape/Sauvola_v3_att_w3.5.7.11.15.19_k0_R0_a0_bnorm/conv_att/Conv2D/Conv2DBackpropInputConv2DBackpropInput?+G3?/~?!?SG3L???0Q      Y@Y?#?M????aq??R??X@qnM??????y??2??y?"?

device?Your program is NOT input-bound because only 1.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?23.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Pascal)(: B 