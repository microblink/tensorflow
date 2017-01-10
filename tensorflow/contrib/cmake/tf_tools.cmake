set(tf_tools_proto_text_src_dir "${tensorflow_source_dir}/tensorflow/tools/proto_text")

file(GLOB tf_tools_srcs
    "${tf_tools_proto_text_src_dir}/gen_proto_text_functions.cc"
    "${tf_tools_proto_text_src_dir}/gen_proto_text_functions_lib.h"
    "${tf_tools_proto_text_src_dir}/gen_proto_text_functions_lib.cc"
)

set(proto_text "proto_text")

if( NOT CMAKE_CROSSCOMPILING )
    if (tensorflow_SEPARATE_STATIC_LIBS)
      add_executable(${proto_text} ${tf_tools_srcs})
      target_link_libraries(${proto_text} PUBLIC ${tensorflow_EXTERNAL_LIBRARIES} tf_protos_cc ${wholearchive_linker_option} tf_core_lib)
    else()
      add_executable(${proto_text}
          ${tf_tools_srcs}
          $<TARGET_OBJECTS:tf_core_lib>
      )

      target_link_libraries(${proto_text} PUBLIC ${tensorflow_EXTERNAL_LIBRARIES} tf_protos_cc)
    endif()

    add_dependencies(${proto_text} tf_core_lib)
    if (tensorflow_ENABLE_GRPC_SUPPORT)
      add_dependencies(${proto_text} grpc)
    endif()

    export( TARGETS ${proto_text} APPEND FILE ${CMAKE_BINARY_DIR}/executables.cmake )
endif()
