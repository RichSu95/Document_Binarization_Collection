	V?F???D@V?F???D@!V?F???D@	?`T?NL???`T?NL??!?`T?NL??"z
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails9V?F???D@???=?$??1>ϟ6?+?@AK?8???\?I}?1Y?o!@Y???;?B??r0*	???x)}?@2f
/Iterator::Root::Prefetch::FlatMap[0]::Generator???v?
@!??eC?oX@)???v?
@1??eC?oX@:Preprocessing2O
Iterator::Root::Prefetch??R%?ޢ?!?%fd??)??R%?ޢ?1?%fd??:Preprocessing2E
Iterator::RootHqh???!h??j?@)????J??1?NO?H???:Preprocessing2X
!Iterator::Root::Prefetch::FlatMap<?_?E?
@!-??DwX@)%Ί??>o?1~c6̼?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?21.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?`T?NL??I?>߷m6@Q???? >S@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???=?$?????=?$??!???=?$??      ??!       "	>ϟ6?+?@>ϟ6?+?@!>ϟ6?+?@*      ??!       2	K?8???\?K?8???\?!K?8???\?:	}?1Y?o!@}?1Y?o!@!}?1Y?o!@B      ??!       J	???;?B?????;?B??!???;?B??R      ??!       Z	???;?B?????;?B??!???;?B??b      ??!       JGPUY?`T?NL??b q?>߷m6@y???? >S@