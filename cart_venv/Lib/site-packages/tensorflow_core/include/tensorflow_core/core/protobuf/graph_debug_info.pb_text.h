// GENERATED FILE - DO NOT MODIFY
#ifndef tensorflow_core_protobuf_graph_debug_info_proto_H_
#define tensorflow_core_protobuf_graph_debug_info_proto_H_

#include "tensorflow/core/protobuf/graph_debug_info.pb.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Message-text conversion for tensorflow.GraphDebugInfo.FileLineCol
string ProtoDebugString(
    const ::tensorflow::GraphDebugInfo_FileLineCol& msg);
string ProtoShortDebugString(
    const ::tensorflow::GraphDebugInfo_FileLineCol& msg);
bool ProtoParseFromString(
    const string& s,
    ::tensorflow::GraphDebugInfo_FileLineCol* msg)
        TF_MUST_USE_RESULT;

// Message-text conversion for tensorflow.GraphDebugInfo.StackTrace
string ProtoDebugString(
    const ::tensorflow::GraphDebugInfo_StackTrace& msg);
string ProtoShortDebugString(
    const ::tensorflow::GraphDebugInfo_StackTrace& msg);
bool ProtoParseFromString(
    const string& s,
    ::tensorflow::GraphDebugInfo_StackTrace* msg)
        TF_MUST_USE_RESULT;

// Message-text conversion for tensorflow.GraphDebugInfo
string ProtoDebugString(
    const ::tensorflow::GraphDebugInfo& msg);
string ProtoShortDebugString(
    const ::tensorflow::GraphDebugInfo& msg);
bool ProtoParseFromString(
    const string& s,
    ::tensorflow::GraphDebugInfo* msg)
        TF_MUST_USE_RESULT;

}  // namespace tensorflow

#endif  // tensorflow_core_protobuf_graph_debug_info_proto_H_
