	Mi?-?8@Mi?-?8@!Mi?-?8@	????F??????F??!????F??"z
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails9Mi?-?8@???e?i??14Փ?G/@A?g	2*\?I??C?!@YA?m??r0*	??? ?ӵ@2f
/Iterator::Root::Prefetch::FlatMap[0]::Generator?vKr?@!|$?4?X@)?vKr?@1|$?4?X@:Preprocessing2E
Iterator::Root??>rkұ?!m?|?Q???)qs* ??1???j"??:Preprocessing2O
Iterator::Root::PrefetchiR
?????!C??9???)iR
?????1C??9???:Preprocessing2X
!Iterator::Root::Prefetch::FlatMapF`?o`@!?B?X@)wJ??l?1?G?[8??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?35.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9????F??I\?=?^-B@Q?q? l?O@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???e?i?????e?i??!???e?i??      ??!       "	4Փ?G/@4Փ?G/@!4Փ?G/@*      ??!       2	?g	2*\??g	2*\?!?g	2*\?:	??C?!@??C?!@!??C?!@B      ??!       J	A?m??A?m??!A?m??R      ??!       Z	A?m??A?m??!A?m??b      ??!       JGPUY????F??b q\?=?^-B@y?q? l?O@