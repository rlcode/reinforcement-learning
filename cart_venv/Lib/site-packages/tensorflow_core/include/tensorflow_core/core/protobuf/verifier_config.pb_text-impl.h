// GENERATED FILE - DO NOT MODIFY
#ifndef tensorflow_core_protobuf_verifier_config_proto_IMPL_H_
#define tensorflow_core_protobuf_verifier_config_proto_IMPL_H_

#include "tensorflow/core/lib/strings/proto_text_util.h"
#include "tensorflow/core/lib/strings/scanner.h"
#include "tensorflow/core/protobuf/verifier_config.pb.h"
#include "tensorflow/core/protobuf/verifier_config.pb_text.h"

namespace tensorflow {

namespace internal {

void AppendProtoDebugString(
    ::tensorflow::strings::ProtoTextOutput* o,
    const ::tensorflow::VerifierConfig& msg);
bool ProtoParseFromScanner(
    ::tensorflow::strings::Scanner* scanner, bool nested, bool close_curly,
    ::tensorflow::VerifierConfig* msg);

}  // namespace internal

}  // namespace tensorflow

#endif  // tensorflow_core_protobuf_verifier_config_proto_IMPL_H_
