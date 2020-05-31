// GENERATED FILE - DO NOT MODIFY
#ifndef tensorflow_core_protobuf_graph_debug_info_proto_IMPL_H_
#define tensorflow_core_protobuf_graph_debug_info_proto_IMPL_H_

#include "tensorflow/core/lib/strings/proto_text_util.h"
#include "tensorflow/core/lib/strings/scanner.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb_text.h"

namespace tensorflow {

namespace internal {

void AppendProtoDebugString(
    ::tensorflow::strings::ProtoTextOutput* o,
    const ::tensorflow::GraphDebugInfo_FileLineCol& msg);
bool ProtoParseFromScanner(
    ::tensorflow::strings::Scanner* scanner, bool nested, bool close_curly,
    ::tensorflow::GraphDebugInfo_FileLineCol* msg);

void AppendProtoDebugString(
    ::tensorflow::strings::ProtoTextOutput* o,
    const ::tensorflow::GraphDebugInfo_StackTrace& msg);
bool ProtoParseFromScanner(
    ::tensorflow::strings::Scanner* scanner, bool nested, bool close_curly,
    ::tensorflow::GraphDebugInfo_StackTrace* msg);

void AppendProtoDebugString(
    ::tensorflow::strings::ProtoTextOutput* o,
    const ::tensorflow::GraphDebugInfo& msg);
bool ProtoParseFromScanner(
    ::tensorflow::strings::Scanner* scanner, bool nested, bool close_curly,
    ::tensorflow::GraphDebugInfo* msg);

}  // namespace internal

}  // namespace tensorflow

#endif  // tensorflow_core_protobuf_graph_debug_info_proto_IMPL_H_
