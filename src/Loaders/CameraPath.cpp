/****************************************************************************/
/* Copyright (c) 2009, Felix Klein, Javor Kalojanov
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
#include "CameraPath.hpp"

std::vector<vec3f> positions;
std::vector<vec3f> rotations;

void readAttribute(std::ifstream &, std::vector<vec3f> &, int);

enum{
    PARSE_TRANSLATE_X,
    PARSE_TRANSLATE_Y,
    PARSE_TRANSLATE_Z,
    PARSE_ROTATE_X,
    PARSE_ROTATE_Y,
    PARSE_ROTATE_Z
};

void readAttribute(
                              std::ifstream &aInputStream,
                              std::vector<vec3f> &aVector,
                              int aIndex)
{

    uint i=0;
    std::string line;
    std::getline(aInputStream, line, ';');

    std::stringstream ss(line);
    std::vector<float> values;
    while(!ss.eof()){
        // getIndex;
        ss >> i;
        // reallocate vector if needed
        while(aVector.size() < i)
        {
            vec3f dummy = vec3f::rep(0.f);
            aVector.push_back(dummy);
        }
        ss >> aVector[i - 1][aIndex];
    }
}

int CameraPath::read(const char* aFileName)
{
    std::ifstream input;
    input.open(aFileName);

    if(!input){
        std::cerr << "Failed to Load CameraPath File.";
        return 1;
    }

    std::string buffer, nodeType;
    int BUFFER_LENGTH = 16000;
    char* cBuffer = new char[BUFFER_LENGTH];
    int o_type;
    while(!input.eof())
    {
        input >> buffer;

        if(buffer == "createNode")
        {

            input >> nodeType;
            //std::cout << buffer << " => create Node - "<< nodeType << std::endl;
            if(nodeType == "animCurveTL"){
                do{
                    input >> buffer;
                }while (buffer != "-n");
                input >> buffer;
                //std::cout << "fix Operate Mode "<< buffer << std::endl;
                if(buffer.find("translateX") != std::string::npos)
                    o_type = PARSE_TRANSLATE_X;
                else if(buffer.find("translateY") != std::string::npos)
                    o_type = PARSE_TRANSLATE_Y;
                else if(buffer.find("translateZ") != std::string::npos)
                    o_type = PARSE_TRANSLATE_Z;
                else if(buffer.find("rotateX") != std::string::npos)
                    o_type = PARSE_ROTATE_X;
                else if(buffer.find("rotateY") != std::string::npos)
                    o_type = PARSE_ROTATE_Y;
                else if(buffer.find("rotateZ") != std::string::npos)
                    o_type = PARSE_ROTATE_Z;
            }
        }

        else if(buffer == "setAttr")
        {

            do{
                input >> buffer;
            }while( buffer.c_str()[0] != '"');

            //std::cout << "set Attribute "<< std::buffer << std::endl;
            if(buffer == "\".ktv[0:3]\""){
                //std::cout << "fill Attribute!"<< std::endl;
                if( o_type == PARSE_TRANSLATE_X)
                    readAttribute(input, positions, 0);
                else if( o_type == PARSE_TRANSLATE_Y)
                    readAttribute(input, positions, 1);
                else if( o_type == PARSE_TRANSLATE_Z)
                    readAttribute(input, positions, 2);
                else if( o_type == PARSE_ROTATE_X)
                    readAttribute(input, rotations, 0);
                else if( o_type == PARSE_ROTATE_Y)
                    readAttribute(input, rotations, 1);
                else if( o_type == PARSE_ROTATE_Z)
                    readAttribute(input, rotations, 2);
            }

        }
        input.getline(cBuffer, BUFFER_LENGTH);
    }

    mNumFrames = positions.size();
    //std::cout << "#Camera Positions: " << mPositions.size() << std::endl;
    //std::cout << "Positions: " << endl;
    //for(int i=0; i< mPositions.size(); i++)
    //    std::cout << mPositions[i][0] << " " << mPositions[i][1] << " " << mPositions[i][2] << std::endl;
    //std::cout << "Rotations: " << endl;
    //    for(int i=0; i< mRotations.size(); i++)
    //        std::cout << mRotations[i][0] << " " << mRotations[i][1] << " " << mRotations[i][2] << std::endl;
    return 0;
}

CameraManager CameraPath::getCamera(size_t aFrameId) const
{
    CameraManager retval;

    retval.setPosition(positions[aFrameId]);

    //Note: y and z coordinates are swapped, because of .obj/.ma incompatibility
    vec3f rotation;
    rotation.x = rotations[aFrameId].x;
    rotation.y = rotations[aFrameId].z;
    rotation.z = rotations[aFrameId].y;
    retval.rotate(rotation);

    return retval;
}

