/**
 * \file
 *
 * CLParameters.hpp
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

#pragma once

#include <string>       // for string
#include <map>          // for multimap
#include <vector>       // for vector

/**
 * Represents command-line parameters.
 * Recognized styles:
 *                      --parameter
 *                      --parameter=value
 *                      -parameter
 *                      -parameter=value
 *                      parameter
 *                      parameter=value
 */
class CLParameters {
public:
    CLParameters(int argc, char *argv[]);
    CLParameters();
    virtual ~CLParameters();

    std::string getParam(const std::string& key) const;
    std::vector<std::string> getParams(const std::string& key) const;
    const std::multimap<std::string, std::string>& getAllParams() const;

    void addParam(const std::string& key, const std::string& value);

    void parse(int argc, char* argv[]);
    
private:
    std::multimap<std::string, std::string> params_;
};

