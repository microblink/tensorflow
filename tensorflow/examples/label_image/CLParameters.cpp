/**
 * \file
 *
 * CLParameters.cpp
 *
 *  Created on: Nov 15, 2012
 *      Author: dodo
 *
 * Copyright (c)2012 Racuni.hr d.o.o. All rights reserved.
 *
 * ANY UNAUTHORIZED USE OR SALE, DUPLICATION, OR DISTRIBUTION
 * OF THIS PROGRAM OR ANY OF ITS PARTS, IN SOURCE OR BINARY FORMS,
 * WITH OR WITHOUT MODIFICATION, WITH THE PURPOSE OF ACQUIRING
 * UNLAWFUL MATERIAL OR ANY OTHER BENEFIT IS PROHIBITED!
 * THIS PROGRAM IS PROTECTED BY COPYRIGHT LAWS AND YOU MAY NOT
 * REVERSE ENGINEER, DECOMPILE, OR DISASSEMBLE IT.
 */

#include <cstddef>  // for size_t
#include <string>    // for operator<, allocator, basic_string, operator!=, operator==
#include <utility>   // for pair

#include "CLParameters.hpp"

CLParameters::CLParameters(int argc, char* argv[]) {
    parse(argc, argv);
}

CLParameters::CLParameters() {}

CLParameters::~CLParameters() {}

std::string CLParameters::getParam(const std::string& key) const {
    if (params_.count(key)) {
        return params_.find(key)->second;
    } else {
        return "";
    }
}

std::vector<std::string> CLParameters::getParams(const std::string& key) const {
    std::vector<std::string> results;
    std::pair<std::multimap<std::string, std::string>::const_iterator,
            std::multimap<std::string, std::string>::const_iterator> ret;
    ret = params_.equal_range(key);
    for (std::multimap<std::string, std::string>::const_iterator it = ret.first; it != ret.second; ++it) {
        results.push_back(it->second);
    }
    return results;
}

const std::multimap<std::string, std::string> &CLParameters::getAllParams() const {
    return params_;
}

void CLParameters::addParam(const std::string& key, const std::string& value) {
    if(!key.empty()) {
        params_.insert(std::pair<std::string, std::string>(key, value));
    }
}

void CLParameters::parse(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string chunk(argv[i]);
        std::string key, value;
        size_t del = chunk.find('=');
        if (del == std::string::npos) {
            key = chunk;
            value = "1";
        } else {
            key = chunk.substr(0, del);
            value = chunk.substr(del + 1);
        }
        // clear -- or - if they exist
        if (key.substr(0, 2) == "--") {
            key = key.substr(2);
        } else if (key[0] == '-') {
            key = key.substr(1);
        }
        if (key != "") {
            params_.insert(std::pair<std::string, std::string>(key, value));
//            params_[key] = value;
//            LOGI("command-line parameter: %s=%s", key.c_str(), value.c_str());
        }
    }
}
