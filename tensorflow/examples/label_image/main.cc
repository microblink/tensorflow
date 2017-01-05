/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// A minimal but useful C++ example showing how to load an Imagenet-style object
// recognition TensorFlow model, prepare input images for it, run them through
// the graph, and interpret the results.
//
// It's designed to have as few dependencies and be as clear as possible, so
// it's more verbose than it could be in production code. In particular, using
// auto for the types of a lot of the returned values from TensorFlow calls can
// remove a lot of boilerplate, but I find the explicit types useful in sample
// code to make it simple to look up the classes involved.
//
// To use it, compile and then run in a working directory with the
// learning/brain/tutorials/label_image/data/ folder below it, and you should
// see the top five labels for the example Lena image output. You can then
// customize it to use your own models or images by changing the file names at
// the top of the main() function.
//
// The googlenet_graph.pb file included by default is created from Inception.

#include <fstream>
#include <vector>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/io_ops.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include "ElapsedTimer.hpp"
#include "CLParameters.hpp"
#include "fast_executor.h"

#include "rapidjson/prettywriter.h"

#ifdef TF_KERNEL_BENCHMARK
#include "tensorflow/core/common_runtime/direct_session.h"
#include "tensorflow/core/common_runtime/threadpool_device.h"
#endif

#ifdef _MSC_VER
#include <Windows.h>
#else
#include <sys/stat.h>       // for stat, mkdir, S_ISDIR
#include <unistd.h>         // for mkstemp, ssize_t, unlink
#include <dirent.h>         // for dirent, closedir, opendir, readdir, DIR
#endif

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
Status ReadLabelsFile(string file_name, std::vector<string>* result,
                      size_t* found_label_count) {
  std::ifstream file(file_name);
  if (!file) {
    return tensorflow::errors::NotFound("Labels file ", file_name,
                                        " not found.");
  }
  result->clear();
  string line;
  while (std::getline(file, line)) {
    result->push_back(line);
  }
  *found_label_count = result->size();
  const int padding = 16;
  while (result->size() % padding) {
    result->emplace_back();
  }
  return Status::OK();
}

// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
Status ReadTensorFromImageFile(string file_name, const bool resize_input_image, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std, const int wanted_channels,
                               std::vector<Tensor>* out_tensors) {
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string input_name = "file_reader";
  string output_name = "normalized";
  auto file_reader = tensorflow::ops::ReadFile(root.WithOpName(input_name), file_name);
  // Now try to figure out what kind of file it is and decode it.
  Output image_reader;
  if (tensorflow::StringPiece(file_name).ends_with(".png")) {
    image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
                             DecodePng::Channels(wanted_channels));
  } else if (tensorflow::StringPiece(file_name).ends_with(".gif")) {
    image_reader = DecodeGif(root.WithOpName("gif_reader"), file_reader);
  } else {
    // Assume if it's neither a PNG nor a GIF then it must be a JPEG.
    image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                              DecodeJpeg::Channels(wanted_channels));
  }
  // Now cast the image data to float so we can do normal math on it.
  auto float_caster =
      Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);
  // The convention for image ops in TensorFlow is that all images are expected
  // to be in batches, so that they're four-dimensional arrays with indices of
  // [batch, height, width, channel]. Because we only have a single image, we
  // have to add a batch dimension of 1 to the start with ExpandDims().
  if ( resize_input_image ) {
      auto dims_expander = ExpandDims(root, float_caster, 0);
      // Bilinearly resize the image to fit the required dimensions.
      auto resized = ResizeBilinear(
          root, dims_expander,
          Const(root.WithOpName("size"), {input_height, input_width}));
      // Subtract the mean and divide by the scale.
      Div(root.WithOpName(output_name), Sub(root, resized, {input_mean}),
          {input_std});
  } else {
      ExpandDims(root.WithOpName( output_name ), float_caster, 0);
  }

  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensor.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(session->Run({}, {output_name}, {}, out_tensors));
  return Status::OK();
}

Status LoadGraph( std::string graph_file_name, std::unique_ptr< tensorflow::FastSession >* session) {
    tensorflow::GraphDef graph_def;

    Status load_graph_status =
        ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if (!load_graph_status.ok()) {
      return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                          graph_file_name, "'");
    }

    session->reset( tensorflow::NewFastSession( graph_def ) );

    return Status::OK();
}

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(string graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  tensorflow::SessionOptions session_options;
  session_options.config.set_log_device_placement( false );
  session_options.config.set_intra_op_parallelism_threads( 8 );
  session_options.config.set_inter_op_parallelism_threads( 1 );

  session->reset( tensorflow::NewSession( session_options ) );
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}

// Analyzes the output of the Inception graph to retrieve the highest scores and
// their positions in the tensor, which correspond to categories.
Status GetTopLabels(const std::vector<Tensor>& outputs, int how_many_labels,
                    Tensor* indices, Tensor* scores) {
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string output_name = "top_k";
  if( outputs.size() < 1 ) {
      LOG( ERROR ) << "Network did not produce a single output!";
      return Status{ tensorflow::error::Code::NOT_FOUND, "Outputs size is 0!" };
  }
  TopKV2(root.WithOpName(output_name), outputs[0], how_many_labels);
  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensors.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  // The TopK node returns two outputs, the scores and their original indices,
  // so we have to append :0 and :1 to specify them both.
  std::vector<Tensor> out_tensors;
  TF_RETURN_IF_ERROR(session->Run({}, {output_name + ":0", output_name + ":1"},
                                  {}, &out_tensors));
  *scores = out_tensors[0];
  *indices = out_tensors[1];
  return Status::OK();
}

// Given the output of a model run, and the name of a file containing the labels
// this prints out the top five highest-scoring values.
Status PrintTopLabels(const std::vector<Tensor>& outputs,
                      string labels_file_name) {
    if ( !labels_file_name.empty() ) {
        std::vector<string> labels;
        size_t label_count;
        Status read_labels_status = ReadLabelsFile(labels_file_name, &labels, &label_count);
        if (!read_labels_status.ok()) {
            LOG(ERROR) << read_labels_status;
            return read_labels_status;
        }
        const int how_many_labels = std::min(5, static_cast<int>(label_count));
        Tensor indices;
        Tensor scores;
        TF_RETURN_IF_ERROR(GetTopLabels(outputs, how_many_labels, &indices, &scores));
        tensorflow::TTypes<float>::Flat scores_flat = scores.flat<float>();
        tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();
        for (int pos = 0; pos < how_many_labels; ++pos) {
            const int label_index = indices_flat(pos);
            const float score = scores_flat(pos);
            if( label_index >= labels.size() ) {
                LOG( INFO ) << "Index outside of labels file: " << label_index << ": " << score;
            } else {
                LOG(INFO) << labels[label_index] << " (" << label_index << "): " << score;
            }
        }
    }
  return Status::OK();
}

// Analyzes the output of the Inception graph to retrieve the highest scores and
// their positions in the tensor, which correspond to categories.
Status GetTopLabels(const Tensor* output, int how_many_labels,
                    Tensor* indices, Tensor* scores) {
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string output_name = "top_k";
  TopKV2(root.WithOpName(output_name), *output, how_many_labels);
  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensors.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  // The TopK node returns two outputs, the scores and their original indices,
  // so we have to append :0 and :1 to specify them both.
  std::vector<Tensor> out_tensors;
  TF_RETURN_IF_ERROR(session->Run({}, {output_name + ":0", output_name + ":1"},
                                  {}, &out_tensors));
  *scores = out_tensors[0];
  *indices = out_tensors[1];
  return Status::OK();
}

// Given the output of a model run, and the name of a file containing the labels
// this prints out the top five highest-scoring values.
Status PrintTopLabels(const Tensor* output,
                      string labels_file_name) {
  std::vector<string> labels;
  size_t label_count;
  Status read_labels_status =
      ReadLabelsFile(labels_file_name, &labels, &label_count);
  if (!read_labels_status.ok()) {
    LOG(ERROR) << read_labels_status;
    return read_labels_status;
  }
  const int how_many_labels = std::min(5, static_cast<int>(label_count));
  Tensor indices;
  Tensor scores;
  TF_RETURN_IF_ERROR(GetTopLabels(output, how_many_labels, &indices, &scores));
  tensorflow::TTypes<float>::Flat scores_flat = scores.flat<float>();
  tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();
  for (int pos = 0; pos < how_many_labels; ++pos) {
    const int label_index = indices_flat(pos);
    const float score = scores_flat(pos);
    LOG(INFO) << labels[label_index] << " (" << label_index << "): " << score;
  }
  return Status::OK();
}

// This is a testing function that returns whether the top label index is the
// one that's expected.
Status CheckTopLabel(const std::vector<Tensor>& outputs, int expected,
                     bool* is_expected) {
  *is_expected = false;
  Tensor indices;
  Tensor scores;
  const int how_many_labels = 1;
  TF_RETURN_IF_ERROR(GetTopLabels(outputs, how_many_labels, &indices, &scores));
  tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();
  if (indices_flat(0) != expected) {
    LOG(ERROR) << "Expected label #" << expected << " but got #"
               << indices_flat(0);
    *is_expected = false;
  } else {
    *is_expected = true;
  }
  return Status::OK();
}

void list_files_in_folder( const std::string& folderName, std::vector<std::string>& files ) {
#if !defined _WIN32 && !defined _WIN64
    DIR* d;
    struct dirent* dirent;
    struct stat fstat;

    d = opendir(folderName.c_str());
    if (!d) {
        LOG( ERROR ) << "Invalid directory " << folderName;
    } else {
        while ((dirent = readdir(d)) != NULL) {
            std::string filename(dirent->d_name);
            if (filename == "." || filename == "..") continue;
            filename = folderName + "/" + filename;
            if (stat(filename.c_str(), &fstat) < 0) continue;
            if (!S_ISDIR(fstat.st_mode)) {
                files.push_back(filename);
            }
        }

        closedir(d);
    }
//#elif WINAPI_FAMILY != WINAPI_FAMILY_PHONE_APP
#elif WINAPI_FAMILY == WINAPI_FAMILY_DESKTOP_APP
    HANDLE hFind = INVALID_HANDLE_VALUE;
    WIN32_FIND_DATA ffd;
    std::stringstream ss;
    ss << folderName << "\\*";

    hFind = FindFirstFile(ss.str().c_str(), &ffd);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (!(ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                std::string filename(ffd.cFileName);
                filename = folderName + "\\" + filename;
                files.push_back(filename);
            }
        } while (FindNextFile(hFind, &ffd) != 0);
    }
    FindClose(hFind);
#else
    LOGE("Not implemented for Windows Phone");
#endif
}

void print_help() {
    printf( "Running: ./tf_benchmark --model=path/to/model.pb \n"
            "                        --img-folder=path/to/images/folder \n"
            "                        --input-layer-name=INPUT_LAYER_NAME \n"
            "                        --output-layer-name=OUTPUT_LAYER_NAME \n"
            "                       [--model-labels=path/to/model-labels.txt] \n"
            "                       [--input-size=widthxheight] \n"
            "                       [--wanted-input-channels=3] \n"
            "                       [--input-mean=0] \n"
            "                       [--input-std=1] \n" );
}

int main(int argc, char* argv[]) {
  CLParameters params( argc, argv );

  bool resize_input_image = false;
  int32 input_mean = 0;
  int32 input_std = 1;

  std::string input_layer = params.getParam( "input-layer-name" );
  std::string output_layer = params.getParam( "output-layer-name" );
  std::string input_dim = params.getParam( "input-size" );
  std::string image_root = params.getParam( "img-folder" );
  std::string labels = params.getParam( "model-labels" );
  std::string graph_path = params.getParam( "model" );
  std::string input_channels_str = params.getParam( "wanted-input-channels" );
  std::string input_mean_str = params.getParam( "input-mean" );
  std::string input_std_str = params.getParam( "input-std" );

  auto required_params = { input_layer, output_layer, image_root, graph_path };
  if( std::any_of( required_params.begin(), required_params.end(), []( const auto& str ) { return str.empty(); } ) ) {
      print_help();
      return 1;
  }

  int32 input_width = -1;
  int32 input_height = -1;

  bool quiet_mode = labels.empty();

  // extract dimensions
  if( !input_dim.empty() ) {
      auto x_pos = input_dim.find( 'x' );
      if ( x_pos == std::string::npos ) {
          LOG( ERROR ) << "input-size must be in format widthxheight";
          return 1;
      }

      auto w_str = input_dim.substr( 0, x_pos );
      auto h_str = input_dim.substr( x_pos + 1 );

      input_width = atoi( w_str.c_str() );
      input_height = atoi( h_str.c_str() );

      resize_input_image = true;
    }

  int wanted_input_channels = 3;
  if ( !input_channels_str.empty() ) {
    wanted_input_channels = atoi( input_channels_str.c_str() );
  }

  if ( !input_mean_str.empty() ) {
      input_mean = atoi( input_mean_str.c_str() );
  }

  if ( !input_std_str.empty() ) {
      input_std = atoi( input_std_str.c_str() );
  }

  // We need to call this to set up global state for TensorFlow.
  tensorflow::port::InitMain(argv[0], &argc, &argv);
//  if (argc > 1) {
//    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
//    return -1;
//  }

  std::unique_ptr<tensorflow::Session> session;
  Status load_graph_status = LoadGraph(graph_path, &session);

//  std::unique_ptr< tensorflow::FastSession > session;
//  Status load_graph_status = LoadGraph( graph_path, &session );


  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return -1;
  }

  std::vector< std::string > images;
  list_files_in_folder( image_root, images );

  std::sort( images.begin(), images.end() );

  ElapsedTimer global_timer;

  double times = 0.0;
  int num = 0;
  bool first = true;

  std::vector< Tensor > final_outputs;
  std::vector< double > image_times;

  for ( const auto& image_path : images ) {
      if( !quiet_mode ) {
        LOG( INFO ) << "Running image " << image_path;
      }

      // Get the image from disk as a float array of numbers, resized and normalized
      // to the specifications the main graph expects.
      std::vector<Tensor> resized_tensors;
      Status read_tensor_status =
          ReadTensorFromImageFile(image_path, resize_input_image, input_height, input_width, input_mean,
                                  input_std, wanted_input_channels, &resized_tensors);
      if (!read_tensor_status.ok()) {
        LOG(ERROR) << read_tensor_status;
        return -1;
      }
      const Tensor& resized_tensor = resized_tensors[0];

      ElapsedTimer local_timer;

      // Actually run the image through the model.
      std::vector<Tensor> outputs;
      Status run_status = session->Run({{input_layer, resized_tensor}},
                                       {output_layer}, {}, &outputs);

      if ( !run_status.ok() ) {
          LOG( ERROR ) << "There was error running the model: " << run_status;
          return 1;
      }

//      Tensor output { tensorflow::DT_FLOAT, tensorflow::TensorShape{ {1, 3} } };
//      Status run_status = session->Run( resized_tensor, &output );

//      float* data = outputs.back().flat<float>().data();

      double tm = local_timer.toc();

      if( !first ) {
          times += tm;
          ++num;
      }

      if( outputs.size() > 0 ) {
        final_outputs.push_back( outputs[ 0 ] );
        image_times.push_back( tm );
      } else {
          LOG( ERROR ) << "Model did not produce a single output!";
      }

      if ( !quiet_mode ) {
        LOG( INFO ) << "Model ran for " << tm << " ms";

          // Do something interesting with the results we've generated.
          Status print_status = PrintTopLabels(outputs, labels);
          if (!print_status.ok()) {
            LOG(ERROR) << "Running print failed: " << print_status;
            return -1;
          }
      }

      first = false;
  }

  double total_time = global_timer.toc();
  double average_per_image = times / static_cast< double >( num );

  if( !quiet_mode ) {
      LOG( INFO ) << "All images processed in " << total_time << " ms";
      LOG( INFO ) << "Average per image: " << average_per_image << " ms";
  }

  rapidjson::StringBuffer buffer;
  rapidjson::PrettyWriter< rapidjson::StringBuffer > writer( buffer );

  writer.StartObject();
  writer.String( "runner" ); writer.String( "tensorflow" );
  writer.String( "model" ); writer.String( graph_path.c_str() );
  writer.String( "input_layer_name" ); writer.String( input_layer.c_str() );
  writer.String( "output_layer_name" ); writer.String( output_layer.c_str() );
  writer.String( "will_resize_images" ); writer.Bool( !input_dim.empty() );
  if( !input_dim.empty() ) {
      writer.String( "images_resized_to" ); writer.String( input_dim.c_str() );
  }
  writer.String( "total_time" ); writer.Double( total_time );
  writer.String( "total_images" ); writer.Int( static_cast< int >( images.size() ) );
  writer.String( "average_per_image" ); writer.Double( average_per_image );

  writer.String( "image_info" );
  writer.StartObject();

  for( auto i = 0U; i < images.size(); ++i ) {
      writer.String( images[ i ].c_str() );
      writer.StartObject();

      writer.String( "time" ); writer.Double( image_times[ i ] );
      writer.String( "output" );
      writer.StartArray();

      auto scores_flat = final_outputs[ i ].flat<float>();
      auto num_elements = scores_flat.size();
      for( auto i = 0U; i < num_elements; ++i ) {
          writer.Double( scores_flat( i ) );
      }

      writer.EndArray();
      writer.EndObject();
  }

  writer.EndObject();


#ifdef TF_KERNEL_BENCHMARK

    writer.String( "average_kernel_times" );
    writer.StartObject();

    // extract kernel times from threadpool_device
    tensorflow::DirectSession* ds = static_cast< tensorflow::DirectSession* >( session.get() );
    auto executors = ds->executors();
    LOG( INFO ) << "There are " << executors.size() << " executors";
    tensorflow::ThreadPoolDevice* tpdev = static_cast< tensorflow::ThreadPoolDevice* >( executors[0]->device() );
    const auto& kernel_times = tpdev->kernel_times();

    for( const auto& pair : kernel_times ) {
        double avg = pair.second / num;
        if( !quiet_mode ) {
            LOG( INFO ) << "Average for kernel " << pair.first << ": " << avg << " ms";
        }
        writer.String( pair.first.c_str() ); writer.Double( avg );
    }

    writer.EndObject();
#endif

    writer.EndObject();

    FILE* f_output = fopen( "tensorflow_bench.json", "wt" );
    fprintf( f_output, "%s\n", buffer.GetString() );
    fclose( f_output );

  return 0;
}
