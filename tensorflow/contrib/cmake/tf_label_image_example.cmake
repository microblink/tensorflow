set(tf_label_image_example_srcs
    "${tensorflow_source_dir}/tensorflow/examples/label_image/main.cc"
    "${tensorflow_source_dir}/tensorflow/examples/label_image/fast_executor.cc"
    "${tensorflow_source_dir}/tensorflow/examples/label_image/ElapsedTimer.cpp"
    "${tensorflow_source_dir}/tensorflow/examples/label_image/DateTime.cpp"
    "${tensorflow_source_dir}/tensorflow/examples/label_image/CLParameters.cpp"
)

if( NOT tensorflow_SEPARATE_STATIC_LIBS )
    add_executable(tf_benchmark
        ${tf_label_image_example_srcs}
        $<TARGET_OBJECTS:tf_core_lib>
        $<TARGET_OBJECTS:tf_core_cpu>
        $<TARGET_OBJECTS:tf_core_framework>
        $<TARGET_OBJECTS:tf_core_kernels>
        $<TARGET_OBJECTS:tf_cc_framework>
        $<TARGET_OBJECTS:tf_cc_ops>
        $<TARGET_OBJECTS:tf_core_ops>
        $<TARGET_OBJECTS:tf_core_direct_session>
        $<$<BOOL:${tensorflow_ENABLE_GPU}>:$<TARGET_OBJECTS:tf_stream_executor>>
    )

    # for some reason, this is not added automatically
    add_dependencies( tf_benchmark tf_core_lib tf_core_cpu tf_core_framework tf_core_kernels tf_cc_framework tf_cc_ops tf_core_ops tf_core_direct_session tf_protos_cc )

else()
    add_executable(tf_benchmark
        ${tf_label_image_example_srcs}
    )
endif()

target_link_libraries(tf_benchmark PUBLIC
    tf_protos_cc
    ${tf_core_gpu_kernels_lib}
    ${tensorflow_EXTERNAL_LIBRARIES}
)
