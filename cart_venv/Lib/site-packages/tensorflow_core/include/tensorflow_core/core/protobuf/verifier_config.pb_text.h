// GENERATED FILE - DO NOT MODIFY
#ifndef tensorflow_core_protobuf_verifier_config_proto_H_
#define tensorflow_core_protobuf_verifier_config_proto_H_

#include "tensorflow/core/protobuf/verifier_config.pb.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Enum text output for tensorflow.VerifierConfig.Toggle
const char* EnumName_VerifierConfig_Toggle(
    ::tensorflow::VerifierConfig_Toggle value);

// Message-text conversion for tensorflow.VerifierConfig
string ProtoDebugString(
    const ::tensorflow::VerifierConfig& msg);
string ProtoShortDebugString(
    const ::tensorflow::VerifierConfig& msg);
bool ProtoParseFromString(
    const string& s,
    ::tensorflow::VerifierConfig* msg)
        TF_MUST_USE_RESULT;

}  // namespace tensorflow

#endif  // tensorflow_core_protobuf_verifier_config_proto_H_
