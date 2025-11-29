// xml_parser.cpp
#include "xml_parser.h"
#include <iostream>
#include <cstring>
#include <libxml/xpath.h>

XMLParser::XMLParser() {
    xmlInitParser(); // Initialize libxml2
    LIBXML_TEST_VERSION
}

XMLParser::~XMLParser() {
    xmlCleanupParser(); // Cleanup libxml2
}

std::vector<TargetInfo> XMLParser::parseFile(const std::string& filename) {
    std::vector<TargetInfo> targets;
    xmlDoc* doc = nullptr;
    xmlXPathContext* xpathCtx = nullptr;
    xmlXPathObject* xpathObj = nullptr;

    doc = xmlParseFile(filename.c_str());
    if (doc == nullptr) {
        std::cerr << "Error: Could not parse file " << filename << std::endl;
        return targets;
    }

    xpathCtx = xmlXPathNewContext(doc);
    if (xpathCtx == nullptr) {
        std::cerr << "Error: Unable to create XPath context" << std::endl;
        xmlFreeDoc(doc);
        return targets;
    }

    // Evaluate XPath expression to find all <target> nodes
    xpathObj = xmlXPathEvalExpression((const xmlChar*)"/targets/target", xpathCtx);
    if (xpathObj == nullptr || xpathObj->nodesetval == nullptr) {
        std::cerr << "Error: No target nodes found or invalid XPath result." << std::endl;
        goto CLEANUP;
    }

    // Iterate through found nodes
    for (int i = 0; i < xpathObj->nodesetval->nodeNr; i++) {
        xmlNode* targetNode = xpathObj->nodesetval->nodeTab[i];
        TargetInfo info;

        // Extract URL and Data using helper function
        info.url = getNodeContent(targetNode, "url");
        info.data = getNodeContent(targetNode, "data");

        if (!info.url.empty()) { // Basic check to avoid empty targets
            targets.push_back(info);
        }
    }

CLEANUP:
    // Free allocated resources
    if (xpathObj) xmlXPathFreeObject(xpathObj);
    if (xpathCtx) xmlXPathFreeContext(xpathCtx);
    if (doc) xmlFreeDoc(doc);
    return targets;
}

std::string XMLParser::getNodeContent(xmlNode* node, const std::string& nodeName) {
    std::string content;
    for (xmlNode* cur = node->children; cur != nullptr; cur = cur->next) {
        if (cur->type == XML_ELEMENT_NODE && xmlStrcmp(cur->name, (const xmlChar*)nodeName.c_str()) == 0) {
            xmlChar* nodeContent = xmlNodeGetContent(cur);
            if (nodeContent) {
                content = std::string((char*)nodeContent);
                xmlFree(nodeContent);
            }
            break;
        }
    }
    return content;
}