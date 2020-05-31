// GENERATED FILE - DO NOT MODIFY
#ifndef tensorflow_core_protobuf_trace_events_proto_H_
#define tensorflow_core_protobuf_trace_events_proto_H_

#include "tensorflow/core/protobuf/trace_events.pb.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace profiler {

// Message-text conversion for tensorflow.profiler.Trace
string ProtoDebugString(
    const ::tensorflow::profiler::Trace& msg);
string ProtoShortDebugString(
    const ::tensorflow::profiler::Trace& msg);
bool ProtoParseFromString(
    const string& s,
    ::tensorflow::profiler::Trace* msg)
        TF_MUST_USE_RESULT;

// Message-text conversion for tensorflow.profiler.Device
string ProtoDebugString(
    const ::tensorflow::profiler::Device& msg);
string ProtoShortDebugString(
    const ::tensorflow::profiler::Device& msg);
bool ProtoParseFromString(
    const string& s,
    ::tensorflow::profiler::Device* msg)
        TF_MUST_USE_RESULT;

// Message-text conversion for tensorflow.profiler.Resource
string ProtoDebugString(
    const ::tensorflow::profiler::Resource& msg);
string ProtoShortDebugString(
    const ::tensorflow::profiler::Resource& msg);
bool ProtoParseFromString(
    const string& s,
    ::tensorflow::profiler::Resource* msg)
        TF_MUST_USE_RESULT;

// Message-text conversion for tensorflow.profiler.TraceEvent
string ProtoDebugString(
    const ::tensorflow::profiler::TraceEvent& msg);
string ProtoShortDebugString(
    const ::tensorflow::profiler::TraceEvent& msg);
bool ProtoParseFromString(
    const string& s,
    ::tensorflow::profiler::TraceEvent* msg)
        TF_MUST_USE_RESULT;

}  // namespace profiler
}  // namespace tensorflow

#endif  // tensorflow_core_protobuf_trace_events_proto_H_
