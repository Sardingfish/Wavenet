#include "Wavenet/Logger.h"

void Logger::_print (std::string cls, std::string fun, std::string type, std::string format, ...) {
    
    std::string col = "\033[1m";
    if      (type == "ERROR")   { col = "\033[1;31m"; }
    else if (type == "WARNING") { col = "\033[1;31m"; }
    else if (type == "INFO")    { col = "\033[1;34m"; }

    std::cout << "\033[1m" << std::left << std::setw(25) << ("<" + cls + "::" + fun + "> ") << col << std::setw(7) << type << "\033[0m ";
    va_list args;
    va_start (args, format);
    vprintf (format.c_str(), args);
    va_end (args);
    std::cout << std::endl;
    return;
}

void Logger::_fctprint (std::string fun, std::string type, std::string format, ...) {
    
    std::string col = "\033[1m";
    if      (type == "ERROR")   { col = "\033[1;31m"; }
    else if (type == "WARNING") { col = "\033[1;31m"; }
    else if (type == "INFO")    { col = "\033[1;34m"; }

    std::cout << "\033[1m" << std::left << std::setw(25) << ("<" + fun + "> ") << col << std::setw(7) << type << "\033[0m ";
    va_list args;
    va_start (args, format);
    vprintf (format.c_str(), args);
    va_end (args);
    std::cout << std::endl;
    return;
}
