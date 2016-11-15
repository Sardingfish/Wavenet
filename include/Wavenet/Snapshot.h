#ifndef WAVENET_SNAPSHOT_H
#define WAVENET_SNAPSHOT_H

/**
 * @file   Snapshot.h
 * @author Andreas Sogaard
 * @date   15 November 2016
 * @brief  Class to write Wavenet objects to file.
 */

// STL include(s).
#include <string> /* std::string */
#include <vector> /* std::vector */
#include <cstdio> /* snprintf */

// Wavenet include(s).
#include "Wavenet/Utilities.h"
#include "Wavenet/Logger.h"


namespace wavenet {

class Wavenet; /* To resolve circular dependence. */

/**
 * Class to write Wavenet objects to file.
 *
 * Allows the user to save a Wavenet object to file, and to load from file. The 
 * snapshot requires a file name or pattern, either in the form of a pure 
 * string, or with one format specifiers which is assumed to indicate a series 
 * of snapshots distinguish by an integer number. In the latter case, the 
 * snapshot furthermore supports increment and decrement operators, meaning that
 * it can easily navigate between successive snapshot. This is convenient when 
 * training with several initialisations in the same base project.
 *
 * If no format specifiers are present in the pattern, the pattern is assummed 
 * to be the file name to which to write.
 */
class Snapshot : public Logger {

public:
    
    /// Constructor(s).
    Snapshot () {};
    Snapshot (const std::string& pattern) :
        m_pattern(pattern)
    {};
    Snapshot (const std::string& pattern, const int& number) :
        m_pattern(pattern), m_number(number)
    {};
    

    /// Destructor(s).
    ~Snapshot () {};
    

    /// Set method(s).
    inline void setPattern (const std::string& pattern) { m_pattern = pattern; return; }
    inline void setNumber  (const int&         number)  { m_number  = number;  return; }
    

    /// Get method(s).
    inline int         number  () const { return m_number; }
    inline std::string pattern () const { return m_pattern; }
    

    /// Directory structure and naming method(s).
    // Check whether the requested file already exists.
    inline bool        exists () const { return fileExists(file()); }
    // Get the file name, constructed from the pattern and possibly the number.
           std::string file   () const;
    

    /// Increment/decrement method(s).
    // Pre-fix increment/decrement: ++snap
    Snapshot& operator++ ();
    Snapshot& operator-- ();
    // Pre-fix increment/decrement: snap++
    Snapshot  operator++ (int);
    Snapshot  operator-- (int);
    
    
protected:

    /// Internal method(s).
    // Utility function to check whether the pattern contains any format 
    // specifiers (%).
    inline bool hasFormatSpecifier () const { return m_pattern.find("%") != std::string::npos; }


private:
    
    /// Data member(s).
    /**
     * The file name pattern used when saving the snapshot. 
     * 
     * May contain one format specifier (%) of the type "...%*{u,d}...", 
     * indicating how to format the (serial) number.
     */
    std::string m_pattern = "";
    /**
     * The (serial) number of the snapshot.
     *
     * If the file name pattern contains a format specifier (%), the snapshot can 
     * navigate between successively numbered snapshot using the increment and 
     * decrement operators.
     */
    int m_number = 0;
    
};

} // namespace

#endif // WAVENET_SNAPSHOT_H
