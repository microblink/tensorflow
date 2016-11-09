set(tf_label_image_example_srcs
    "${tensorflow_source_dir}/tensorflow/examples/label_image/main.cc"
    "${tensorflow_source_dir}/tensorflow/examples/label_image/ElapsedTimer.cpp"
    "${tensorflow_source_dir}/tensorflow/examples/label_image/fast_executor.cc"
)

if (tensorflow_SEPARATE_STATIC_LIBS)
  add_executable( tf_label_image_example EXCLUDE_FROM_ALL ${tf_label_image_example_srcs} )
  target_link_libraries( tf_label_image_example
    ${wholearchive_linker_option}
    tf_cc_framework
    tf_cc_ops
    tf_core_cpu
    tf_core_framework
    tf_core_kernels
    tf_core_lib
    tf_core_ops
    tf_core_direct_session
    tf_protos_cc
    ${tensorflow_EXTERNAL_LIBRARIES}
  )
  add_dependencies( tf_label_image_example ${tensorflow_EXTERNAL_DEPENDENCIES} )
  link_directories( "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}" )
else()
  add_executable( tf_label_image_example EXCLUDE_FROM_ALL
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

  target_link_libraries(tf_label_image_example PUBLIC
      tf_protos_cc
      #${tf_core_gpu_kernels_lib}
      ${tensorflow_EXTERNAL_LIBRARIES}
  )
endif()
