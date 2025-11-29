// xml_parser.h
#ifndef XML_PARSER_H
#define XML_PARSER_H

#include <string>
#include <vector>
#include <libxml/parser.h>
#include <libxml/tree.h>

// Structure to hold extracted target data
struct TargetInfo {
    std::string url;
    std::string data;
};

class XMLParser {
public:
    XMLParser();
    ~XMLParser();

    // Parses XML from a file and returns a vector of TargetInfo
    std::vector<TargetInfo> parseFile(const std::string& filename);

private:
    // Helper function to extract content from an xmlNode
    std::string getNodeContent(xmlNode* node, const std::string& nodeName);
};

#endif