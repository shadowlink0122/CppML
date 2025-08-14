#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <streambuf>
#include <memory>
#include <stdexcept>
#include <cassert>
#include <cmath>
#include <vector>

/**
 * @file test_utils.hpp
 * @brief Utility functions and classes for unit testing
 */

namespace MLLib {
namespace test {

/**
 * @class OutputCapture
 * @brief Captures standard output and error streams for testing
 * 
 * This class temporarily redirects stdout and stderr to temporary files
 * to prevent test output from appearing in the console, which is useful
 * for both unit tests and integration tests.
 */
class OutputCapture {
public:
    /**
     * @brief Constructor - starts capturing output
     */
    OutputCapture();
    
    /**
     * @brief Destructor - restores original streams
     */
    ~OutputCapture();
    
    /**
     * @brief Get captured stdout content
     * @return String containing captured stdout
     */
    std::string getCapturedStdout() const;
    
    /**
     * @brief Get captured stderr content
     * @return String containing captured stderr
     */
    std::string getCapturedStderr() const;
    
    /**
     * @brief Clear captured content
     */
    void clear();
    
    // Disable copy constructor and assignment operator
    OutputCapture(const OutputCapture&) = delete;
    OutputCapture& operator=(const OutputCapture&) = delete;

private:
    std::streambuf* original_cout_;
    std::streambuf* original_cerr_;
    std::ostringstream captured_cout_;
    std::ostringstream captured_stderr_;
};

/**
 * @class TestCase
 * @brief Base class for test cases with assertion utilities
 */
class TestCase {
public:
    /**
     * @brief Constructor
     * @param name Test case name
     */
    explicit TestCase(const std::string& name);
    
    /**
     * @brief Destructor
     */
    virtual ~TestCase() = default;
    
    /**
     * @brief Run the test case
     * @return true if all tests pass, false otherwise
     */
    bool run();
    
    /**
     * @brief Get test case name
     * @return Test case name
     */
    const std::string& getName() const { return name_; }
    
    /**
     * @brief Get number of passed tests
     * @return Number of passed tests
     */
    int getPassedCount() const { return passed_count_; }
    
    /**
     * @brief Get number of failed tests
     * @return Number of failed tests
     */
    int getFailedCount() const { return failed_count_; }

protected:
    /**
     * @brief Test implementation - to be overridden by derived classes
     */
    virtual void test() = 0;
    
    /**
     * @brief Assert that condition is true
     * @param condition Condition to check
     * @param message Error message if assertion fails
     */
    void assertTrue(bool condition, const std::string& message = "");
    
    /**
     * @brief Assert that condition is false
     * @param condition Condition to check
     * @param message Error message if assertion fails
     */
    void assertFalse(bool condition, const std::string& message = "");
    
    /**
     * @brief Assert that two values are equal
     * @param expected Expected value
     * @param actual Actual value
     * @param message Error message if assertion fails
     */
template<typename T>
void assertEqual(const T& expected, const T& actual, const std::string& message);

// Specialized template declarations for non-numeric types
template<>
void assertEqual<std::string>(const std::string& expected, const std::string& actual, const std::string& message);    /**
     * @brief Assert that two values are not equal
     * @param not_expected Value that should not match
     * @param actual Actual value
     * @param message Error message if assertion fails
     */
    template<typename T>
    void assertNotEqual(const T& not_expected, const T& actual, const std::string& message = "");
    
    /**
     * @brief Assert that two floating point values are approximately equal
     * @param expected Expected value
     * @param actual Actual value
     * @param tolerance Tolerance for comparison
     * @param message Error message if assertion fails
     */
    void assertNear(double expected, double actual, double tolerance = 1e-6, const std::string& message = "");
    
    /**
     * @brief Assert that a pointer is not null
     * @param ptr Pointer to check
     * @param message Error message if assertion fails
     */
    template<typename T>
    void assertNotNull(T* ptr, const std::string& message = "");
    
    /**
     * @brief Assert that a pointer is null
     * @param ptr Pointer to check
     * @param message Error message if assertion fails
     */
    template<typename T>
    void assertNull(T* ptr, const std::string& message = "");
    
    /**
     * @brief Assert that vectors are equal
     * @param expected Expected vector
     * @param actual Actual vector
     * @param message Error message if assertion fails
     */
    template<typename T>
    void assertVectorEqual(const std::vector<T>& expected, const std::vector<T>& actual, const std::string& message = "");
    
    /**
     * @brief Assert that vectors are approximately equal (for floating point)
     * @param expected Expected vector
     * @param actual Actual vector
     * @param tolerance Tolerance for comparison
     * @param message Error message if assertion fails
     */
    void assertVectorNear(const std::vector<double>& expected, const std::vector<double>& actual, 
                         double tolerance = 1e-6, const std::string& message = "");
    
    /**
     * @brief Assert that an exception is thrown
     * @param func Function that should throw
     * @param message Error message if no exception is thrown
     */
    template<typename ExceptionType, typename Func>
    void assertThrows(Func func, const std::string& message = "");
    
    /**
     * @brief Assert that no exception is thrown
     * @param func Function that should not throw
     * @param message Error message if exception is thrown
     */
    template<typename Func>
    void assertNoThrow(Func func, const std::string& message = "");

private:
    std::string name_;
    int passed_count_;
    int failed_count_;
    
    /**
     * @brief Record assertion result
     * @param condition Result of assertion
     * @param message Error message
     */
    void recordAssertion(bool condition, const std::string& message);
};

/**
 * @class TestSuite
 * @brief Collection of test cases with reporting
 */
class TestSuite {
public:
    /**
     * @brief Constructor
     * @param name Test suite name
     */
    explicit TestSuite(const std::string& name);
    
    /**
     * @brief Add test case to suite
     * @param test_case Test case to add
     */
    void addTest(std::unique_ptr<TestCase> test_case);
    
    /**
     * @brief Run all test cases in the suite
     * @return true if all tests pass, false otherwise
     */
    bool runAll();
    
    /**
     * @brief Get test suite name
     * @return Test suite name
     */
    const std::string& getName() const { return name_; }
    
    /**
     * @brief Get total number of passed tests
     * @return Number of passed tests
     */
    int getTotalPassedCount() const;
    
    /**
     * @brief Get total number of failed tests
     * @return Number of failed tests
     */
    int getTotalFailedCount() const;

private:
    std::string name_;
    std::vector<std::unique_ptr<TestCase>> test_cases_;
};

/**
 * @brief Create temporary file for testing
 * @param content Content to write to file
 * @return Temporary file path
 */
std::string createTempFile(const std::string& content = "");

/**
 * @brief Remove temporary file
 * @param filepath Path to temporary file
 */
void removeTempFile(const std::string& filepath);

/**
 * @brief Create temporary directory for testing
 * @return Temporary directory path
 */
std::string createTempDirectory();

/**
 * @brief Remove temporary directory and its contents
 * @param dirpath Path to temporary directory
 */
void removeTempDirectory(const std::string& dirpath);

/**
 * @brief Check if file exists
 * @param filepath Path to file
 * @return true if file exists, false otherwise
 */
bool fileExists(const std::string& filepath);

/**
 * @brief Read entire file content
 * @param filepath Path to file
 * @return File content as string
 */
std::string readFileContent(const std::string& filepath);


// Template implementations

template<typename T>
void MLLib::test::TestCase::assertEqual(const T& expected, const T& actual, const std::string& message) {
    bool condition = (expected == actual);
    std::string full_message = message;
    if (!full_message.empty()) {
        full_message += " - ";
    }
    full_message += "Expected: " + std::to_string(expected) + ", Actual: " + std::to_string(actual);
    recordAssertion(condition, full_message);
}

template<typename T>
void MLLib::test::TestCase::assertNotEqual(const T& not_expected, const T& actual, const std::string& message) {
    bool condition = (not_expected != actual);
    std::string full_message = message;
    if (!full_message.empty()) {
        full_message += " - ";
    }
    full_message += "Values should not be equal: " + std::to_string(actual);
    recordAssertion(condition, full_message);
}

template<typename T>
void MLLib::test::TestCase::assertNotNull(T* ptr, const std::string& message) {
    bool condition = (ptr != nullptr);
    std::string full_message = message;
    if (!full_message.empty()) {
        full_message += " - ";
    }
    full_message += "Pointer should not be null";
    recordAssertion(condition, full_message);
}

template<typename T>
void MLLib::test::TestCase::assertNull(T* ptr, const std::string& message) {
    bool condition = (ptr == nullptr);
    std::string full_message = message;
    if (!full_message.empty()) {
        full_message += " - ";
    }
    full_message += "Pointer should be null";
    recordAssertion(condition, full_message);
}

template<typename T>
void MLLib::test::TestCase::assertVectorEqual(const std::vector<T>& expected, const std::vector<T>& actual, const std::string& message) {
    bool condition = (expected.size() == actual.size());
    if (condition) {
        for (size_t i = 0; i < expected.size(); ++i) {
            if (expected[i] != actual[i]) {
                condition = false;
                break;
            }
        }
    }
    std::string full_message = message;
    if (!full_message.empty()) {
        full_message += " - ";
    }
    full_message += "Vector comparison failed";
    recordAssertion(condition, full_message);
}

template<typename ExceptionType, typename Func>
void MLLib::test::TestCase::assertThrows(Func func, const std::string& message) {
    bool caught_expected_exception = false;
    try {
        func();
    } catch (const ExceptionType&) {
        caught_expected_exception = true;
    } catch (...) {
        // Wrong exception type
    }
    
    std::string full_message = message;
    if (!full_message.empty()) {
        full_message += " - ";
    }
    full_message += "Expected exception was not thrown";
    recordAssertion(caught_expected_exception, full_message);
}

template<typename Func>
void MLLib::test::TestCase::assertNoThrow(Func func, const std::string& message) {
    bool no_exception = true;
    try {
        func();
    } catch (...) {
        no_exception = false;
    }
    
    std::string full_message = message;
    if (!full_message.empty()) {
        full_message += " - ";
    }
    full_message += "No exception should be thrown";
    recordAssertion(no_exception, full_message);
}

// File system helper functions for testing
std::string createTempDirectory();
void removeTempDirectory(const std::string& path);
bool fileExists(const std::string& path);
bool directoryExists(const std::string& path);

} // namespace test
} // namespace MLLib
