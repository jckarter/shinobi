/****************************************************************************/
/* Copyright (c) 2009, Javor Kalojanov
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*/
/****************************************************************************/

#include "StdAfx.hpp"
#include "LightSourceLoader.hpp"

AreaLightSource LightSourceLoader::loadFromFile(const char* aFileName)
{
    AreaLightSource retval;

    retval.position.x = retval.position.y = retval.position.z = 0.f;
    retval.normal.x = retval.normal.y = retval.normal.z = 0.f;
    retval.intensity.x = retval.intensity.y = retval.intensity.z = 0.f;
    retval.edge1.x = retval.edge1.y = retval.edge1.z = 0.f;
    retval.edge2.x = retval.edge2.y = retval.edge2.z = 0.f;


    std::ifstream input;
    input.open(aFileName);

    if(!input)
        std::cerr << "Could not open file " << aFileName << " for reading.\n"
        << __FILE__ << __LINE__ << std::endl;

    std::string line, buff;
 
    while ( !input.eof() )
    {
        std::getline(input, line);

        while(line.find_first_of('#', 0) != std::string::npos)//eat comments
        {
            std::getline(input, line);
        }

        std::stringstream ss(line);
        ss >> buff;

        if (buff == "position")
        {
            ss >> retval.position.x >> retval.position.y >> retval.position.z;
        } 
        else if (buff == "normal")
        {
            ss >> retval.normal.x >> retval.normal.y >> retval.normal.z;
        } 
        else if (buff == "intensity")
        {
            ss >> retval.intensity.x >> retval.intensity.y >> retval.intensity.z;
        }
        else if (buff == "edge1")
        {
            ss >> retval.edge1.x >> retval.edge1.y >> retval.edge1.z;
        }
        else if (buff == "edge2")
        {
            ss >> retval.edge2.x >> retval.edge2.y >> retval.edge2.z;
        }
    }

    if (retval.normal.x == 0.f && retval.normal.y == 0.f &&
        retval.normal.z == 0.f)
    {
        retval.normal = ~(retval.edge1 % retval.edge2);
    }

    input.close();

    return retval;
}
