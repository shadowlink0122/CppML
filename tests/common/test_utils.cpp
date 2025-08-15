#include "test_utils.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>

namespace MLLib {
namespace test {

// OutputCapture implementation
OutputCapture::OutputCapture()
    : original_cout_(std::cout.rdbuf()), original_cerr_(std::cerr.rdbuf()) {
  // Redirect cout and cerr to our string streams
  std::cout.rdbuf(captured_cout_.rdbuf());
  std::cerr.rdbuf(captured_stderr_.rdbuf());
}

// Template specializations for non-numeric types
template <>
void TestCase::assertEqual<std::string>(const std::string& expected,
                                        const std::string& actual,
                                        const std::string& message) {
  bool condition = (expected == actual);
  std::string full_message = message;
  if (!full_message.empty()) {
    full_message += " - ";
  }
  full_message += "Expected: \"" + expected + "\", Actual: \"" + actual + "\"";
  recordAssertion(condition, full_message);
}

OutputCapture::~OutputCapture() {
  // Restore original streams
  std::cout.rdbuf(original_cout_);
  std::cerr.rdbuf(original_cerr_);
}

std::string OutputCapture::getCapturedStdout() const {
  return captured_cout_.str();
}

std::string OutputCapture::getCapturedStderr() const {
  return captured_stderr_.str();
}

void OutputCapture::clear() {
  captured_cout_.str("");
  captured_cout_.clear();
  captured_stderr_.str("");
  captured_stderr_.clear();
}

// TestCase implementation
TestCase::TestCase(const std::string& name)
    : name_(name), passed_count_(0), failed_count_(0), execution_time_ms_(0.0) {
}

bool TestCase::run() {
  passed_count_ = 0;
  failed_count_ = 0;

  std::cout << "Running test: " << name_ << std::endl;

  // Start timing
  auto start_time = std::chrono::high_resolution_clock::now();

  try {
    // Capture output during test execution
    OutputCapture capture;
    test();

    // Note: captured output is automatically discarded
    // This ensures test output doesn't interfere with test reporting

  } catch (const std::exception& e) {
    std::cout << "Test " << name_ << " threw unexpected exception: " << e.what()
              << std::endl;
    failed_count_++;
    return false;
  } catch (...) {
    std::cout << "Test " << name_ << " threw unexpected unknown exception"
              << std::endl;
    failed_count_++;
    return false;
  }

  // End timing
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);
  execution_time_ms_ = duration.count() / 1000.0;

  if (failed_count_ == 0) {
    std::cout << "✅ " << name_ << " PASSED (" << passed_count_
              << " assertions, " << std::fixed << std::setprecision(2)
              << execution_time_ms_ << "ms)" << std::endl;
    return true;
  } else {
    std::cout << "❌ " << name_ << " FAILED (" << failed_count_ << " failed, "
              << passed_count_ << " passed, " << std::fixed
              << std::setprecision(2) << execution_time_ms_ << "ms)"
              << std::endl;
    return false;
  }
}

void TestCase::assertTrue(bool condition, const std::string& message) {
  std::string full_message = message;
  if (!full_message.empty()) {
    full_message += " - ";
  }
  full_message += "Expected true";
  recordAssertion(condition, full_message);
}

void TestCase::assertFalse(bool condition, const std::string& message) {
  std::string full_message = message;
  if (!full_message.empty()) {
    full_message += " - ";
  }
  full_message += "Expected false";
  recordAssertion(!condition, full_message);
}

void TestCase::assertNear(double expected, double actual, double tolerance,
                          const std::string& message) {
  bool condition = std::abs(expected - actual) <= tolerance;
  std::string full_message = message;
  if (!full_message.empty()) {
    full_message += " - ";
  }
  full_message += "Expected: " + std::to_string(expected) +
                  ", Actual: " + std::to_string(actual) +
                  ", Tolerance: " + std::to_string(tolerance);
  recordAssertion(condition, full_message);
}

void TestCase::assertVectorNear(const std::vector<double>& expected,
                                const std::vector<double>& actual,
                                double tolerance, const std::string& message) {
  bool condition = (expected.size() == actual.size());
  if (condition) {
    for (size_t i = 0; i < expected.size(); ++i) {
      if (std::abs(expected[i] - actual[i]) > tolerance) {
        condition = false;
        break;
      }
    }
  }
  std::string full_message = message;
  if (!full_message.empty()) {
    full_message += " - ";
  }
  full_message += "Vector near comparison failed with tolerance " +
                  std::to_string(tolerance);
  recordAssertion(condition, full_message);
}

void TestCase::recordAssertion(bool condition, const std::string& message) {
  if (condition) {
    passed_count_++;
  } else {
    failed_count_++;
    std::cout << "  ASSERTION FAILED: " << message << std::endl;
  }
}

// TestSuite implementation
TestSuite::TestSuite(const std::string& name) : name_(name) {}

void TestSuite::addTest(std::unique_ptr<TestCase> test_case) {
  test_cases_.push_back(std::move(test_case));
}

bool TestSuite::runAll() {
  std::cout << "\n=== Running Test Suite: " << name_ << " ===" << std::endl;

  bool all_passed = true;
  int suite_passed = 0;
  int suite_failed = 0;

  for (auto& test_case : test_cases_) {
    bool result = test_case->run();
    if (result) {
      suite_passed++;
    } else {
      suite_failed++;
      all_passed = false;
    }
  }

  std::cout << "\n=== Test Suite Results: " << name_ << " ===" << std::endl;
  std::cout << "Total tests: " << test_cases_.size() << std::endl;
  std::cout << "Passed: " << suite_passed << std::endl;
  std::cout << "Failed: " << suite_failed << std::endl;
  std::cout << "Total assertions: "
            << getTotalPassedCount() + getTotalFailedCount() << std::endl;
  std::cout << "Passed assertions: " << getTotalPassedCount() << std::endl;
  std::cout << "Failed assertions: " << getTotalFailedCount() << std::endl;

  if (all_passed) {
    std::cout << "✅ ALL TESTS PASSED" << std::endl;
  } else {
    std::cout << "❌ SOME TESTS FAILED" << std::endl;
  }

  return all_passed;
}

int TestSuite::getTotalPassedCount() const {
  int total = 0;
  for (const auto& test_case : test_cases_) {
    total += test_case->getPassedCount();
  }
  return total;
}

int TestSuite::getTotalFailedCount() const {
  int total = 0;
  for (const auto& test_case : test_cases_) {
    total += test_case->getFailedCount();
  }
  return total;
}

// Utility functions
std::string createTempFile(const std::string& content) {
  std::string temp_path = "/tmp/mllib_test_" + std::to_string(std::rand());
  std::ofstream file(temp_path);
  if (file.is_open()) {
    file << content;
    file.close();
  }
  return temp_path;
}

void removeTempFile(const std::string& filepath) {
  std::remove(filepath.c_str());
}

std::string readFileContent(const std::string& filepath) {
  std::ifstream file(filepath);
  if (!file.is_open()) {
    return "";
  }

  std::ostringstream content;
  content << file.rdbuf();
  return content.str();
}

// File system helper functions implementation
std::string createTempDirectory() {
  std::string temp_dir = "/tmp/mllib_test_dir_" + std::to_string(std::rand());
  mkdir(temp_dir.c_str(), 0755);
  return temp_dir;
}

void removeTempDirectory(const std::string& path) {
  if (access(path.c_str(), F_OK) == 0) {
    std::string cmd = "rm -rf " + path;
    system(cmd.c_str());
  }
}

bool fileExists(const std::string& path) {
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0) && S_ISREG(buffer.st_mode);
}

bool directoryExists(const std::string& path) {
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0) && S_ISDIR(buffer.st_mode);
}

}  // namespace test
}  // namespace MLLib
