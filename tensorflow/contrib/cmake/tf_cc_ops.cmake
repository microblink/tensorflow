########################################################
# tf_cc_framework library
########################################################
set(tf_cc_framework_srcs
    "${tensorflow_source_dir}/tensorflow/cc/framework/ops.h"
    "${tensorflow_source_dir}/tensorflow/cc/framework/ops.cc"
    "${tensorflow_source_dir}/tensorflow/cc/framework/scope.h"
    "${tensorflow_source_dir}/tensorflow/cc/framework/scope.cc"
)

add_library(tf_cc_framework ${TF_OBJECTLIB} ${tf_cc_framework_srcs})

add_dependencies(tf_cc_framework tf_core_framework)

########################################################
# tf_cc_op_gen_main library
########################################################
set(tf_cc_op_gen_main_srcs
    "${tensorflow_source_dir}/tensorflow/cc/framework/cc_op_gen.cc"
    "${tensorflow_source_dir}/tensorflow/cc/framework/cc_op_gen_main.cc"
    "${tensorflow_source_dir}/tensorflow/cc/framework/cc_op_gen.h"
)
if ( NOT tensorflow_SEPARATE_STATIC_LIBS )
  add_library(tf_cc_op_gen_main ${TF_OBJECTLIB} ${tf_cc_op_gen_main_srcs})
  add_dependencies(tf_cc_op_gen_main tf_core_framework)
endif()


########################################################
# tf_gen_op_wrapper_cc executables
########################################################

# create directory for ops generated files
set(cc_ops_target_dir ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/cc/ops)

add_custom_target(create_cc_ops_header_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${cc_ops_target_dir}
)

set(tf_cc_ops_generated_files)

set(tf_cc_op_lib_names
    ${tf_op_lib_names}
    "user_ops"
)
foreach(tf_cc_op_lib_name ${tf_cc_op_lib_names})
  if ( tensorflow_SEPARATE_STATIC_LIBS )
    add_executable(${tf_cc_op_lib_name}_gen_cc ${tf_cc_op_gen_main_srcs})
    target_link_libraries(${tf_cc_op_lib_name}_gen_cc PRIVATE
        ${wholearchive_linker_option}
        tf_${tf_cc_op_lib_name}
        tf_core_lib
        tf_core_framework
        tf_protos_cc
        ${tensorflow_EXTERNAL_LIBRARIES}
    )
  else()
    # Using <TARGET_OBJECTS:...> to work around an issue where no ops were
    # registered (static initializers dropped by the linker because the ops
    # are not used explicitly in the *_gen_cc executables).
    add_executable(${tf_cc_op_lib_name}_gen_cc
        $<TARGET_OBJECTS:tf_cc_op_gen_main>
        $<TARGET_OBJECTS:tf_${tf_cc_op_lib_name}>
        $<TARGET_OBJECTS:tf_core_lib>
        $<TARGET_OBJECTS:tf_core_framework>
    )

    target_link_libraries(${tf_cc_op_lib_name}_gen_cc PRIVATE
        tf_protos_cc
        ${tensorflow_EXTERNAL_LIBRARIES}
    )
  endif()

  set(cc_ops_include_internal 0)
  if(${tf_cc_op_lib_name} STREQUAL "sendrecv_ops")
      set(cc_ops_include_internal 1)
  endif()

  add_custom_command(
      OUTPUT ${cc_ops_target_dir}/${tf_cc_op_lib_name}.h
             ${cc_ops_target_dir}/${tf_cc_op_lib_name}.cc
      COMMAND ${tf_cc_op_lib_name}_gen_cc ${cc_ops_target_dir}/${tf_cc_op_lib_name}.h ${cc_ops_target_dir}/${tf_cc_op_lib_name}.cc ${cc_ops_include_internal}
      DEPENDS ${tf_cc_op_lib_name}_gen_cc create_cc_ops_header_dir
  )

  list(APPEND tf_cc_ops_generated_files ${cc_ops_target_dir}/${tf_cc_op_lib_name}.h)
  list(APPEND tf_cc_ops_generated_files ${cc_ops_target_dir}/${tf_cc_op_lib_name}.cc)
endforeach()


########################################################
# tf_cc_ops library
########################################################
add_library(tf_cc_ops ${TF_OBJECTLIB}
    ${tf_cc_ops_generated_files}
    "${tensorflow_source_dir}/tensorflow/cc/ops/const_op.h"
    "${tensorflow_source_dir}/tensorflow/cc/ops/const_op.cc"
    "${tensorflow_source_dir}/tensorflow/cc/ops/standard_ops.h"
)
