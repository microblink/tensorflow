########################################################
# tf_core_distributed_runtime library
########################################################
file(GLOB_RECURSE tf_core_distributed_runtime_srcs
   "${tensorflow_source_dir}/tensorflow/core/distributed_runtime/*.h"
   "${tensorflow_source_dir}/tensorflow/core/distributed_runtime/*.cc"
)

file(GLOB_RECURSE tf_core_distributed_runtime_exclude_srcs
    "${tensorflow_source_dir}/tensorflow/core/distributed_runtime/*test*.h"
    "${tensorflow_source_dir}/tensorflow/core/distributed_runtime/*test*.cc"
    "${tensorflow_source_dir}/tensorflow/core/distributed_runtime/rpc/grpc_tensorflow_server.cc"
)

list(REMOVE_ITEM tf_core_distributed_runtime_srcs ${tf_core_distributed_runtime_exclude_srcs})

add_library(tf_core_distributed_runtime ${TF_OBJECTLIB} ${tf_core_distributed_runtime_srcs})

add_dependencies(tf_core_distributed_runtime
    tf_core_cpu grpc
)

########################################################
# grpc_tensorflow_server executable
########################################################
set(grpc_tensorflow_server_srcs
    "${tensorflow_source_dir}/tensorflow/core/distributed_runtime/rpc/grpc_tensorflow_server.cc"
)

if( NOT tensorflow_SEPARATE_STATIC_LIBS )
    list( APPEND grpc_tensorflow_server_srcs
        $<TARGET_OBJECTS:tf_core_lib>
        $<TARGET_OBJECTS:tf_core_cpu>
        $<TARGET_OBJECTS:tf_core_framework>
        $<TARGET_OBJECTS:tf_core_kernels>
        $<TARGET_OBJECTS:tf_cc_framework>
        $<TARGET_OBJECTS:tf_cc_ops>
        $<TARGET_OBJECTS:tf_core_ops>
        $<TARGET_OBJECTS:tf_core_direct_session>
        $<TARGET_OBJECTS:tf_core_distributed_runtime>
        $<$<BOOL:${tensorflow_ENABLE_GPU}>:$<TARGET_OBJECTS:tf_stream_executor>>
    )
endif()

add_executable(grpc_tensorflow_server
    ${grpc_tensorflow_server_srcs}
)

if( tensorflow_SEPARATE_STATIC_LIBS )
    target_link_libraries(grpc_tensorflow_server PUBLIC
        ${wholearchive_linker_option}
        tf_core_lib
        tf_core_cpu
        tf_core_framework
        tf_core_kernels
        tf_cc_framework
        tf_cc_ops
        tf_core_ops
        tf_core_direct_session
        tf_core_distributed_runtime
        $<$<BOOL:${tensorflow_ENABLE_GPU}>:tf_stream_executor>
    )
endif()

target_link_libraries(grpc_tensorflow_server PUBLIC
    tf_protos_cc
    ${tf_core_gpu_kernels_lib}
    ${tensorflow_EXTERNAL_LIBRARIES}
)
