	?ɨ2??7@?ɨ2??7@!?ɨ2??7@      ??!       "h
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
	T?qs*??T?qs*??!T?qs*??      ??!       "	G?ŧ ?/@G?ŧ ?/@!G?ŧ ?/@*      ??!       2      ??!       :	0G????@0G????@!0G????@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qҚ??@@y?????P@