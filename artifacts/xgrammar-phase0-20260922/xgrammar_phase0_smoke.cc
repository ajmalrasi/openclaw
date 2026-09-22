#include <xgrammar/xgrammar.h>

#include <dlpack/dlpack.h>
#include <picojson.h>
#include <sys/resource.h>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string ReadFile(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open " + path);
  return std::string(std::istreambuf_iterator<char>(input), {});
}

picojson::value ParseJSON(const std::string& path) {
  picojson::value value;
  const std::string error = picojson::parse(value, ReadFile(path));
  if (!error.empty()) throw std::runtime_error(path + ": " + error);
  return value;
}

std::vector<std::string> ReadStringArray(const std::string& path) {
  const auto value = ParseJSON(path);
  std::vector<std::string> result;
  result.reserve(value.get<picojson::array>().size());
  for (const auto& item : value.get<picojson::array>()) result.push_back(item.get<std::string>());
  return result;
}

std::vector<int> ReadTokenFixture(const std::string& path, const std::string& name) {
  const auto root = ParseJSON(path).get<picojson::object>();
  std::vector<int> result;
  for (const auto& item : root.at(name).get<picojson::array>()) {
    result.push_back(static_cast<int>(item.get<double>()));
  }
  return result;
}

bool Feed(xgrammar::GrammarMatcher* matcher, const std::vector<int>& tokens, int* rejected_at) {
  for (size_t index = 0; index < tokens.size(); ++index) {
    if (!matcher->AcceptToken(tokens[index])) {
      *rejected_at = static_cast<int>(index);
      return false;
    }
  }
  *rejected_at = -1;
  return true;
}

void CompileCase(xgrammar::GrammarCompiler* compiler, const char* name, const char* schema) {
  const auto start = std::chrono::steady_clock::now();
  try {
    auto grammar = compiler->CompileJSONSchema(schema, true, std::nullopt, std::nullopt, true);
    const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - start)
                            .count();
    std::cout << "SCHEMA_CASE name=" << name << " result=accepted compile_ms=" << millis
              << " memory_bytes=" << grammar.MemorySizeBytes() << "\n";
  } catch (const std::exception& error) {
    const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - start)
                            .count();
    std::cout << "SCHEMA_CASE name=" << name << " result=rejected compile_ms=" << millis
              << " error=" << error.what() << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: xgrammar_phase0_smoke FIXTURE_DIR\n";
    return 2;
  }
  const std::string root = argv[1];
  const auto vocab = ReadStringArray(root + "/encoded_vocab.json");
  const std::string backend = ReadFile(root + "/backend_tokenizer.json");
  const std::string metadata = xgrammar::TokenizerInfo::DetectMetadataFromHF(backend);
  picojson::value metadata_value;
  const std::string metadata_error = picojson::parse(metadata_value, metadata);
  if (!metadata_error.empty()) throw std::runtime_error("tokenizer metadata: " + metadata_error);
  const auto metadata_object = metadata_value.get<picojson::object>();
  const auto vocab_type = static_cast<xgrammar::VocabType>(
      static_cast<int>(metadata_object.at("vocab_type").get<int64_t>()));
  const bool add_prefix_space = metadata_object.at("add_prefix_space").get<bool>();
  xgrammar::TokenizerInfo tokenizer(
      vocab, vocab_type, static_cast<int>(vocab.size()), std::vector<int32_t>{248046},
      add_prefix_space);

  std::cout << "TOKENIZER vocab_entries=" << vocab.size()
            << " xgrammar_vocab_size=" << tokenizer.GetVocabSize()
            << " metadata=" << metadata << "\n";

  xgrammar::GrammarCompiler compiler(tokenizer, 2, true, 64 * 1024 * 1024);
  const char* primary_schema = R"JSON({
    "type":"object",
    "properties":{
      "name":{"type":"string"},
      "age":{"type":"integer"},
      "tags":{"type":"array","items":{"type":"string"},"maxItems":4}
    },
    "required":["name","age","tags"],
    "additionalProperties":false
  })JSON";
  auto compiled = compiler.CompileJSONSchema(primary_schema, true, std::nullopt, std::nullopt, true);
  xgrammar::GrammarMatcher valid_matcher(compiled, std::vector<int>{248046}, true);
  xgrammar::GrammarMatcher invalid_matcher(compiled, std::vector<int>{248046}, true);

  const auto valid = ReadTokenFixture(root + "/fixture_tokens.json", "valid_primary");
  const auto invalid = ReadTokenFixture(root + "/fixture_tokens.json", "invalid_primary");
  int valid_rejected_at = -1;
  int invalid_rejected_at = -1;
  const bool valid_accepted = Feed(&valid_matcher, valid, &valid_rejected_at);
  const bool invalid_accepted = Feed(&invalid_matcher, invalid, &invalid_rejected_at);
  std::cout << "MATCH valid_accepted=" << valid_accepted
            << " valid_completed=" << valid_matcher.IsCompleted()
            << " valid_rejected_at=" << valid_rejected_at
            << " invalid_accepted=" << invalid_accepted
            << " invalid_rejected_at=" << invalid_rejected_at << "\n";

  const int64_t bitmask_shape[1] = {xgrammar::GetBitmaskSize(tokenizer.GetVocabSize())};
  std::vector<int32_t> bitmask(bitmask_shape[0]);
  DLTensor tensor{};
  tensor.data = bitmask.data();
  tensor.device = DLDevice{kDLCPU, 0};
  tensor.ndim = 1;
  tensor.dtype = xgrammar::GetBitmaskDLType();
  tensor.shape = const_cast<int64_t*>(bitmask_shape);
  tensor.strides = nullptr;
  tensor.byte_offset = 0;
  xgrammar::GrammarMatcher initial_matcher(compiled, std::vector<int>{248046}, true);
  const bool mask_required = initial_matcher.FillNextTokenBitmask(&tensor);
  const int first_token = valid.front();
  const bool first_allowed = (bitmask[first_token / 32] & (1U << (first_token % 32))) != 0;
  std::cout << "BITMASK required=" << mask_required << " words=" << bitmask.size()
            << " bytes=" << bitmask.size() * sizeof(int32_t)
            << " valid_first_token=" << first_token << " first_allowed=" << first_allowed << "\n";

  CompileCase(&compiler, "nested_enum_unicode", R"JSON({
    "type":"object","properties":{
      "status":{"enum":["ready","é","😊"]},
      "items":{"type":"array","items":{"type":"integer"},"minItems":1,"maxItems":8},
      "nested":{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"]}
    },"required":["status","items"],"additionalProperties":false
  })JSON");
  CompileCase(&compiler, "unions_const", R"JSON({
    "type":"object","properties":{
      "kind":{"const":"record"},"value":{"anyOf":[{"type":"number"},{"type":"null"}]}
    },"required":["kind","value"],"additionalProperties":false
  })JSON");
  CompileCase(&compiler, "string_pattern", R"JSON({"type":"string","pattern":"^[A-Z]{2}[0-9]{3}$"})JSON");
  CompileCase(&compiler, "numeric_range", R"JSON({"type":"integer","minimum":1,"maximum":10,"multipleOf":2})JSON");
  CompileCase(&compiler, "external_ref", R"JSON({"$ref":"https://example.com/schema.json"})JSON");

  const bool pass = valid_accepted && valid_matcher.IsCompleted() && !invalid_accepted &&
                    mask_required && first_allowed && tokenizer.GetVocabSize() == 248320;
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
  std::cout << "PHASE0_GATE passed=" << pass
            << " compiler_cache_bytes=" << compiler.GetCacheSizeBytes()
            << " process_max_rss_kib=" << usage.ru_maxrss << "\n";
  return pass ? 0 : 1;
}
