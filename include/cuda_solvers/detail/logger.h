#pragma once
#include <iostream>
#include <unistd.h>

inline bool stderr_is_terminal() {
  return isatty(fileno(stderr));
}

#define GREEN  (stderr_is_terminal() ? "\033[32m" : "")
#define YELLOW (stderr_is_terminal() ? "\033[33m" : "")
#define RED    (stderr_is_terminal() ? "\033[31m" : "")
#define RESET  (stderr_is_terminal() ? "\033[0m"  : "")

#define LOG_INFO(msg) \
  std::cerr << GREEN << "[INFO]" << RESET << " " << msg << '\n'

#define LOG_WARN(msg) \
  std::cerr << YELLOW << "[WARNING]" << RESET << " " << msg << '\n'

#define LOG_ERROR(msg) \
  std::cerr << RED << "[ERROR]" << RESET << " " << msg << '\n'
