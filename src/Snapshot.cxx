#include "Wavenet/Snapshot.h"
#include "Wavenet/Wavenet.h" /* To resolve circular dependence. */

namespace wavenet {
    
std::string Snapshot::file () const {

    // If pattern is a pure file name, without any format specifiers (%), return
    // it.
    if (!hasFormatSpecifier()) { return m_pattern; }
    
    // Otherwise use the member number to complete pattern.
    char buff[100];
    snprintf(buff, sizeof(buff), m_pattern.c_str(), m_number);
    std::string filename = buff;
    return filename;
    
}

Snapshot& Snapshot::operator++ () { 
    if (!hasFormatSpecifier()) { 
        WARNING("Calling pre-fix increment operator for snapshot with no format specifiers in file name pattern");
    } 
    ++m_number; 
    return *this;
}

Snapshot& Snapshot::operator-- () { 
    if (!hasFormatSpecifier()) { 
        WARNING("Calling pre-fix decrement operator for snapshot with no format specifiers in file name pattern");
    } 
    --m_number; 
    return *this;
}

Snapshot Snapshot::operator++ (int) { 
    if (!hasFormatSpecifier()) { 
        WARNING("Calling post-fix increment operator for snapshot with no format specifiers in file name pattern");
    } 
    Snapshot tmp(*this); 
    m_number++; 
    return tmp;
}

Snapshot Snapshot::operator-- (int) { 
    if (!hasFormatSpecifier()) { 
        WARNING("Calling post-fix decrement operator for snapshot with no format specifiers in file name pattern");
    } 
    Snapshot tmp(*this); 
    m_number--; 
    return tmp;
}

Snapshot& operator<< (Snapshot& snap, const Wavenet& wavenet) {
     
    // Initialise output stream.
    ofstream stream (snap.file());

    // Stream Wavenet data members out.
    stream << wavenet.m_lambda           << "\n";
    stream << wavenet.m_alpha            << "\n";
    stream << wavenet.m_inertia          << "\n";
    stream << wavenet.m_inertiaTimeScale << "\n";
    stream << wavenet.m_filter           << "\n#\n";
    stream << wavenet.m_momentum         << "\n#\n";

    stream << wavenet.m_batchSize << "\n";
    stream << "BATCHQUEUE" << "\n";
    for (const auto& q : wavenet.m_batchQueue) { stream << q << "\n#\n"; }

    stream << "FILTERLOG" << "\n";
    for (const auto& f : wavenet.m_filterLog)  { stream << f << "\n#\n"; }

    stream << "COSTLOG" << "\n";
    for (const auto& c : wavenet.m_costLog)    { stream << c << "\n"; }

    // Close output stream.
    stream.close();

    return snap;
}

const Snapshot& operator>> (const Snapshot& snap, Wavenet& wavenet) {

    // To stream in values and check for delimeters.
    std::string tmp; 

    // Initialise input stream.
    ifstream stream (snap.file());

    // Stream Wavenet data members int.
    stream >> wavenet.m_lambda;
    stream >> wavenet.m_alpha;
    stream >> wavenet.m_inertia;
    stream >> wavenet.m_inertiaTimeScale;
    
    // Read filter.
    std::vector<double> vec_filter;
    while (stream >> tmp && tmp.find("#") == std::string::npos) {
        try {
            vec_filter.push_back( stod(tmp) );
        } catch (const std::invalid_argument& ia) {;}
    }
    wavenet.m_filter = arma::conv_to< arma::Col<double> >::from(vec_filter);
    
    // Read momentum.
    std::vector<double> vec_momentum;
    while (stream >> tmp && tmp.find("#") == std::string::npos) {
        try {
            vec_momentum.push_back( stod(tmp) );
        } catch (const std::invalid_argument& ia) {;}
    }
    wavenet.m_momentum = arma::conv_to< arma::Col<double> >::from(vec_momentum);
    
    stream >> wavenet.m_batchSize;
    
    // Read batch queue.
    stream >> tmp;
    wavenet.m_batchQueue.clear();
    while (tmp.find("FILTERLOG") == std::string::npos) {
        std::vector<double> vec_momentum;
        while (stream >> tmp) {
            try {
                vec_momentum.push_back( stod(tmp) );
            } catch (const std::invalid_argument& ia) { break; }
        }
        if (!vec_momentum.size()) { break; }
        wavenet.m_batchQueue.push_back( arma::conv_to< arma::Col<double> >::from(vec_momentum) );
    }
    
    // Read filter log.
    wavenet.m_filterLog.clear();
    while (tmp.find("COSTLOG") == std::string::npos) {
        std::vector<double> vec_filter;
        while (stream >> tmp) {
            try {
                vec_filter.push_back( stod(tmp) );
            } catch (const std::invalid_argument& ia) { break; }
        }
        if (!vec_filter.size()) { break; }
        wavenet.m_filterLog.push_back( arma::conv_to< arma::Col<double> >::from(vec_filter) );
    }
    
    // Read cost log.
    wavenet.m_costLog.clear();
    while (!stream.fail()) {
        while (stream >> tmp) {
            try {
                wavenet.m_costLog.push_back( stod(tmp) );
            } catch (const std::invalid_argument& ia) { break; }
        }
    }
    
    // Close the input stream.
    stream.close();

    return snap;
}

} // namespace
