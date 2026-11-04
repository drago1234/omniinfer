// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <execinfo.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

#include <unistd.h>

namespace ox {

enum class LogLevel : unsigned char {
    kDebug = 0,
    kInfo = 1,
    kWarn = 2,
    kError = 3,
};

inline constexpr const char *kDefaultLogPath = "/data/ox_log/ox.log";
inline constexpr std::size_t kMaximumMessageBytes = 8192;
inline constexpr std::size_t kMaximumBacktraceFrames = 64;

// Temporary hard limit for one native log file; it is truncated before further writes at this size.
inline constexpr std::uintmax_t kMaximumLogFileBytes = 500ULL * 1024ULL * 1024ULL;


// Converts the internal enum to the stable level text used in log records.
inline const char *log_level_name(LogLevel level)
{
    switch (level) {
    case LogLevel::kDebug:
        return "DEBUG";
    case LogLevel::kInfo:
        return "INFO";
    case LogLevel::kWarn:
        return "WARN";
    case LogLevel::kError:
        return "ERROR";
    }
    return "INFO";
}


// Lightweight, thread-safe OX native logger that centralizes runtime configuration and all sinks.
class Logger {
public:

    // Returns the process-wide logger singleton and defers runtime configuration until first use.
    static Logger &instance()
    {
        static Logger logger;
        return logger;
    }


    // Checks whether a level is enabled before the caller formats a message.
    bool should_log(LogLevel level)
    {
        ensure_initialized();
        return static_cast<unsigned char>(level)
            >= static_cast<unsigned char>(minimum_level_.load(std::memory_order_relaxed));
    }


    // Commits one complete log record. Exception handlers explicitly capture
    // a backtrace; an ordinary ERROR record does not imply an exception.
    void log(LogLevel level, const std::string &component, const std::string &message)
    {
        if (!should_log(level)) {
            return;
        }

        if (level != LogLevel::kError) {
            std::lock_guard<std::mutex> lock(output_mutex_);
            emit_locked(level, component, message, "");
            return;
        }

        const std::string error_id = std::to_string(::getpid()) + "-"
            + std::to_string(next_error_id_.fetch_add(1, std::memory_order_relaxed));
        std::lock_guard<std::mutex> lock(output_mutex_);
        emit_locked(LogLevel::kError, component, message, error_id);
    }


    // Writes non-runtime text such as CLI help through the same terminal and file sinks.
    void write_cli(const std::string &text)
    {
        ensure_initialized();
        std::lock_guard<std::mutex> lock(output_mutex_);
        std::cout << text;
        std::cout.flush();
        if (file_sink_.is_open() && truncate_file_if_needed_locked(text.size())) {
            file_sink_ << text;
            file_sink_.flush();
            if (!file_sink_) {
                disable_file_sink_locked("write to log file failed; continuing with terminal output only");
            }
        }
    }


    // Resets internal state for isolated environment-driven tests; production code must not call this.
    void reset_for_testing()
    {
        std::lock_guard<std::mutex> init_lock(init_mutex_);
        std::lock_guard<std::mutex> output_lock(output_mutex_);
        file_sink_.close();
        file_sink_.clear();
        file_path_.clear();
        initialized_.store(false, std::memory_order_release);
        next_error_id_.store(1, std::memory_order_relaxed);
    }

private:

    // Keeps construction private so every caller shares the same sinks and locks.
    Logger() = default;


    // Performs one-time, double-checked initialization so each log does not reread the environment.
    void ensure_initialized()
    {
        if (initialized_.load(std::memory_order_acquire)) {
            return;
        }
        std::lock_guard<std::mutex> init_lock(init_mutex_);
        if (initialized_.load(std::memory_order_relaxed)) {
            return;
        }

        std::lock_guard<std::mutex> output_lock(output_mutex_);
        initialize_locked();
        initialized_.store(true, std::memory_order_release);
    }


    // Normalizes environment values to uppercase to make log levels case-insensitive.
    static std::string uppercase(std::string value)
    {
        for (char &ch : value) {
            ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
        }
        return value;
    }


    // Parses log level, file-sink settings, and path, falling back to safe defaults for invalid configuration.
    void initialize_locked()
    {
        const char *raw_level = std::getenv("OMNIINFER_OX_LOG_LEVEL");
        const std::string level_value = raw_level == nullptr ? "" : uppercase(raw_level);
        if (level_value.empty() || level_value == "INFO") {
            minimum_level_.store(LogLevel::kInfo, std::memory_order_relaxed);
        } else if (level_value == "DEBUG") {
            minimum_level_.store(LogLevel::kDebug, std::memory_order_relaxed);
        } else if (level_value == "WARN" || level_value == "WARNING") {
            minimum_level_.store(LogLevel::kWarn, std::memory_order_relaxed);
        } else if (level_value == "ERROR") {
            minimum_level_.store(LogLevel::kError, std::memory_order_relaxed);
        } else {
            minimum_level_.store(LogLevel::kInfo, std::memory_order_relaxed);
            emit_configuration_warning_locked(
                "invalid OMNIINFER_OX_LOG_LEVEL=" + level_value + "; using INFO");
        }

        const char *raw_file = std::getenv("OMNIINFER_OX_LOG_FILE");
        const std::string file_value = raw_file == nullptr ? "" : raw_file;
        if (file_value == "1") {
            const char *raw_path = std::getenv("OMNIINFER_OX_LOG_PATH");
            const std::string path = raw_path == nullptr || std::string(raw_path).empty()
                ? kDefaultLogPath
                : raw_path;
            open_file_sink_locked(path);
        } else if (!file_value.empty() && file_value != "0") {
            emit_configuration_warning_locked(
                "invalid OMNIINFER_OX_LOG_FILE=" + file_value + "; using 0");
        }
    }


    // Creates parent directories, opens the log file in append mode, and retains its path; on failure it keeps terminal-only logging.
    void open_file_sink_locked(const std::string &path)
    {
        std::error_code ec;
        const std::filesystem::path file_path(path);
        const std::filesystem::path parent = file_path.parent_path();
        if (!parent.empty() && !std::filesystem::exists(parent, ec)) {
            std::filesystem::create_directories(parent, ec);
        }
        if (ec || (!parent.empty() && !std::filesystem::is_directory(parent, ec))) {
            emit_configuration_warning_locked(
                "cannot create log directory for " + path + "; continuing with terminal output only");
            return;
        }

        file_sink_.open(path, std::ios::out | std::ios::app);
        if (!file_sink_.is_open()) {
            emit_configuration_warning_locked(
                "cannot open log file " + path + "; continuing with terminal output only");
            return;
        }
        file_path_ = path;
    }


    // Clears the file before the next write would exceed 500 MiB; checks actual size to observe external appends.
    bool truncate_file_if_needed_locked(std::size_t pending_bytes)
    {
        if (!file_sink_.is_open() || file_path_.empty()) {
            return false;
        }

        std::error_code ec;
        const std::uintmax_t current_bytes = std::filesystem::file_size(file_path_, ec);
        if (ec) {
            disable_file_sink_locked(
                "cannot inspect log file " + file_path_ + "; continuing with terminal output only");
            return false;
        }
        const std::uintmax_t pending = static_cast<std::uintmax_t>(pending_bytes);
        if (current_bytes < kMaximumLogFileBytes
            && pending <= kMaximumLogFileBytes - current_bytes) {
            return true;
        }

        file_sink_.close();
        file_sink_.clear();
        file_sink_.open(file_path_, std::ios::out | std::ios::trunc);
        if (!file_sink_.is_open()) {
            disable_file_sink_locked(
                "cannot truncate log file " + file_path_ + "; continuing with terminal output only");
            return false;
        }
        file_sink_.close();
        file_sink_.clear();
        file_sink_.open(file_path_, std::ios::out | std::ios::app);
        if (!file_sink_.is_open()) {
            disable_file_sink_locked(
                "cannot reopen log file " + file_path_ + "; continuing with terminal output only");
            return false;
        }
        emit_configuration_warning_locked(
            "log file reached 500 MiB and was truncated: " + file_path_);
        return true;
    }


    // Disables a failed file sink and reports the downgrade once on the terminal.
    void disable_file_sink_locked(const std::string &reason)
    {
        file_sink_.close();
        file_sink_.clear();
        file_path_.clear();
        emit_configuration_warning_locked(reason);
    }


    // Produces an ISO-8601 timestamp with milliseconds and the local UTC offset.
    static std::string timestamp()
    {
        const auto now = std::chrono::system_clock::now();
        const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        const std::time_t seconds = std::chrono::system_clock::to_time_t(now);
        std::tm local_time {};
        ::localtime_r(&seconds, &local_time);

        std::ostringstream stream;
        stream << std::put_time(&local_time, "%Y-%m-%dT%H:%M:%S")
               << '.' << std::setfill('0') << std::setw(3) << millis.count();
#if defined(__linux__)
        const long offset = local_time.tm_gmtoff;
#else
        const long offset = 0;
#endif
        const char sign = offset < 0 ? '-' : '+';
        const long absolute_offset = offset < 0 ? -offset : offset;
        stream << sign << std::setfill('0') << std::setw(2) << absolute_offset / 3600
               << ':' << std::setw(2) << (absolute_offset % 3600) / 60;
        return stream.str();
    }


    // Escapes control characters and bounds field length so external input cannot break one-line log records.
    static std::string escape_and_limit(const std::string &value)
    {
        std::string escaped;
        escaped.reserve(value.size());
        constexpr char hex[] = "0123456789ABCDEF";
        for (const unsigned char ch : value) {
            if (ch == '\n') {
                escaped += "\\n";
            } else if (ch == '\r') {
                escaped += "\\r";
            } else if (ch == '\t') {
                escaped += "\\t";
            } else if (ch < 0x20 || ch == 0x7f) {
                escaped += "\\x";
                escaped += hex[(ch >> 4) & 0x0f];
                escaped += hex[ch & 0x0f];
            } else {
                escaped += static_cast<char>(ch);
            }
            if (escaped.size() >= kMaximumMessageBytes) {
                escaped.resize(kMaximumMessageBytes);
                escaped += "...[truncated]";
                break;
            }
        }
        return escaped;
    }


    // Builds the stable log-line format shared by every sink.
    std::string format_line(
        LogLevel level, const std::string &component, const std::string &message,
        const std::string &error_id) const
    {
        std::ostringstream stream;
        stream << timestamp() << ' ' << log_level_name(level) << ' '
               << escape_and_limit(component) << " pid=" << ::getpid()
               << " tid=" << std::hash<std::thread::id>{}(std::this_thread::get_id());
        if (!error_id.empty()) {
            stream << " error_id=" << error_id;
        }
        stream << ' ' << escape_and_limit(message);
        return stream.str();
    }


    // While holding the output lock, atomically writes one complete record to terminal and file sinks.
    void emit_locked(
        LogLevel level, const std::string &component, const std::string &message,
        const std::string &error_id)
    {
        const std::string line = format_line(level, component, message, error_id);
        std::cout << line << std::endl;
        if (file_sink_.is_open() && truncate_file_if_needed_locked(line.size() + 1)) {
            file_sink_ << line << std::endl;
            if (!file_sink_) {
                disable_file_sink_locked("write to log file failed; continuing with terminal output only");
            }
        }
    }


    // Emits a configuration warning independently of log level so invalid setup cannot fail silently.
    void emit_configuration_warning_locked(const std::string &message)
    {
        std::cout << format_line(LogLevel::kWarn, "logger", message, "") << std::endl;
    }


    std::atomic<LogLevel> minimum_level_{LogLevel::kInfo};
    std::atomic<bool> initialized_{false};
    std::atomic<unsigned long long> next_error_id_{1};
    std::mutex init_mutex_;
    std::mutex output_mutex_;
    std::ofstream file_sink_;
    std::string file_path_;
};


// RAII stream object behind OX_LOG_* that commits a record when the C++ statement ends.
class LogMessage {
public:

    // Stores a stream log's level and component; the destructor commits it at statement end.
    LogMessage(LogLevel level, const char *component)
        : level_(level), component_(component == nullptr ? "unknown" : component) {}


    // Ensures logging failures never alter request processing or shutdown flow.
    ~LogMessage() noexcept
    {
        try {
            Logger::instance().log(level_, component_, stream_.str());
        } catch (...) {
            // Logging must never alter the request or shutdown control flow.
        }
    }


    // Accepts any ostream-writable value while building the stream log message.
    template <typename T>
    LogMessage &operator<<(const T &value)
    {
        stream_ << value;
        return *this;
    }


    // Preserves compatibility with ostream manipulators such as std::endl.
    LogMessage &operator<<(std::ostream &(*manipulator)(std::ostream &))
    {
        stream_ << manipulator;
        return *this;
    }

private:
    LogLevel level_;
    std::string component_;
    std::ostringstream stream_;
};

}  // namespace ox


// These conditional macros avoid argument evaluation, stringstream construction, and sink locking for disabled levels.
#define OX_LOG_DEBUG(component) \
    if (!::ox::Logger::instance().should_log(::ox::LogLevel::kDebug)) {} else ::ox::LogMessage(::ox::LogLevel::kDebug, (component))
#define OX_LOG_INFO(component) \
    if (!::ox::Logger::instance().should_log(::ox::LogLevel::kInfo)) {} else ::ox::LogMessage(::ox::LogLevel::kInfo, (component))
#define OX_LOG_WARN(component) \
    if (!::ox::Logger::instance().should_log(::ox::LogLevel::kWarn)) {} else ::ox::LogMessage(::ox::LogLevel::kWarn, (component))
#define OX_LOG_ERROR(component) \
    if (!::ox::Logger::instance().should_log(::ox::LogLevel::kError)) {} else ::ox::LogMessage(::ox::LogLevel::kError, (component))

namespace ox {

// Call from an exception handler after its ERROR record. Every frame is an
// ERROR record so Python forwards the full trace, including under the default
// INFO native threshold. The shared trace_id keeps concurrent traces distinct.
inline void print_ox_backtrace(const char *component)
{
    static std::atomic<unsigned long long> next_trace_id {1};
    const std::string trace_id = std::to_string(::getpid()) + "-"
        + std::to_string(next_trace_id.fetch_add(1, std::memory_order_relaxed));
    std::array<void *, kMaximumBacktraceFrames> frames {};
    const int frame_count = ::backtrace(frames.data(), static_cast<int>(frames.size()));
    OX_LOG_ERROR(component) << "OX backtrace trace_id=" << trace_id
        << " frames=" << (frame_count > 0 ? frame_count - 1 : 0);
    if (frame_count <= 1) {
        OX_LOG_ERROR(component) << "OX backtrace trace_id=" << trace_id
            << " capture failed: no caller frames";
        return;
    }
    char **symbols = ::backtrace_symbols(frames.data(), frame_count);
    if (symbols == nullptr) {
        OX_LOG_ERROR(component) << "OX backtrace trace_id=" << trace_id
            << " symbolization failed";
        return;
    }
    // Frame zero is this helper; frame one is the exception handler.
    for (int index = 1; index < frame_count; ++index) {
        OX_LOG_ERROR(component) << "OX backtrace trace_id=" << trace_id
            << " #" << (index - 1) << " " << symbols[index];
    }
    std::free(symbols);
}

}  // namespace ox
