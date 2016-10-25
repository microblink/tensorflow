########################################################
# tf_models_word2vec_ops library
########################################################
file(GLOB tf_models_word2vec_ops_srcs
    "${tensorflow_source_dir}/tensorflow/models/embedding/word2vec_ops.cc"
)

add_library(tf_models_word2vec_ops ${TF_OBJECTLIB} ${tf_models_word2vec_ops_srcs})

add_dependencies(tf_models_word2vec_ops tf_core_framework)

########################################################
# tf_models_word2vec_kernels library
########################################################
file(GLOB tf_models_word2vec_kernels_srcs
    "${tensorflow_source_dir}/tensorflow/models/embedding/word2vec_kernels.cc"
)

add_library(tf_models_word2vec_kernels ${TF_OBJECTLIB} ${tf_models_word2vec_kernels_srcs})

add_dependencies(tf_models_word2vec_kernels tf_core_cpu)
