	!?> ??E@!?> ??E@!!?> ??E@	?H7?7????H7?7???!?H7?7???"q
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
	?e??S9???e??S9??!?e??S9??      ??!       "	??SV???@??SV???@!??SV???@*      ??!       2      ??!       :	?I?5??$@?I?5??$@!?I?5??$@B      ??!       J	??????????????!???????R      ??!       Z	??????????????!???????b      ??!       JGPUY?H7?7???b q4?????8@y????hR@