set(tf_tutorials_example_trainer_srcs
    "${tensorflow_source_dir}/tensorflow/cc/tutorials/example_trainer.cc"
)

if (tensorflow_SEPARATE_STATIC_LIBS)
  add_executable(tf_tutorials_example_trainer EXCLUDE_FROM_ALL ${tf_tutorials_example_trainer_srcs})

  target_link_libraries(tf_tutorials_example_trainer PUBLIC
      tf_core_lib
      tf_core_cpu
      tf_core_framework
      tf_core_kernels
      tf_cc_framework
      tf_cc_ops
      tf_core_ops
      tf_core_direct_session
      $<$<BOOL:${tensorflow_ENABLE_GPU}>:tf_stream_executor>
      tf_protos_cc
      ${tf_core_gpu_kernels_lib}
      ${tensorflow_EXTERNAL_LIBRARIES}
  )
else()
  add_executable(tf_tutorials_example_trainer EXCLUDE_FROM_ALL
      ${tf_tutorials_example_trainer_srcs}
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

  target_link_libraries(tf_tutorials_example_trainer PUBLIC
      tf_protos_cc
      ${tf_core_gpu_kernels_lib}
      ${tensorflow_EXTERNAL_LIBRARIES}
  )
endif()