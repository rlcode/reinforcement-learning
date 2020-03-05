// GENERATED FILE - DO NOT MODIFY
#ifndef tensorflow_core_protobuf_trace_events_proto_IMPL_H_
#define tensorflow_core_protobuf_trace_events_proto_IMPL_H_

#include "tensorflow/core/lib/strings/proto_text_util.h"
#include "tensorflow/core/lib/strings/scanner.h"
#include "tensorflow/core/protobuf/trace_events.pb.h"
#include "tensorflow/core/protobuf/trace_events.pb_text.h"

namespace tensorflow {
namespace profiler {

namespace internal {

void AppendProtoDebugString(
    ::tensorflow::strings::ProtoTextOutput* o,
    const ::tensorflow::profiler::Trace& msg);
bool ProtoParseFromScanner(
    ::tensorflow::strings::Scanner* scanner, bool nested, bool close_curly,
    ::tensorflow::profiler::Trace* msg);

void AppendProtoDebugString(
    ::tensorflow::strings::ProtoTextOutput* o,
    const ::tensorflow::profiler::Device& msg);
bool ProtoParseFromScanner(
    ::tensorflow::strings::Scanner* scanner, bool nested, bool close_curly,
    ::tensorflow::profiler::Device* msg);

void AppendProtoDebugString(
    ::tensorflow::strings::ProtoTextOutput* o,
    const ::tensorflow::profiler::Resource& msg);
bool ProtoParseFromScanner(
    ::tensorflow::strings::Scanner* scanner, bool nested, bool close_curly,
    ::tensorflow::profiler::Resource* msg);

void AppendProtoDebugString(
    ::tensorflow::strings::ProtoTextOutput* o,
    const ::tensorflow::profiler::TraceEvent& msg);
bool ProtoParseFromScanner(
    ::tensorflow::strings::Scanner* scanner, bool nested, bool close_curly,
    ::tensorflow::profiler::TraceEvent* msg);

}  // namespace internal

}  // namespace profiler
}  // namespace tensorflow

#endif  // tensorflow_core_protobuf_trace_events_proto_IMPL_H_
