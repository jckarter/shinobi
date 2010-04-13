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
#include "CameraManager.hpp"



//Performs rotation around all three axes with the argument angles
void CameraManager::rotate (const vec3f& aRotation)
{
    float angleX = aRotation.x * static_cast<float>(M_PI) / 180.f;
    float sinPhyX = sinf(angleX);
    float cosPhyX = cosf(angleX);

    float angleY = aRotation.y * static_cast<float>(M_PI) / 180.f;
    float sinTethaY = sinf(angleY);
    float cosTethaY = cosf(angleY);

    float angleZ = aRotation.z * static_cast<float>(M_PI) / 180.f;
    float sinPsyZ = sinf(angleZ);
    float cosPsyZ = cosf(angleZ);

    vec3f column1;
    column1.x = cosTethaY * cosPsyZ;
    column1.y = sinPhyX * sinTethaY * cosPsyZ - cosPhyX * sinPsyZ;
    column1.z = cosPhyX * sinTethaY * cosPsyZ + sinPhyX * sinPsyZ;

    vec3f column2;
    column2.x = cosTethaY * sinPsyZ;
    column2.y = sinPhyX * sinTethaY * sinPsyZ + cosPhyX * cosPsyZ;
    column2.z = sinPhyX * sinTethaY * sinPsyZ - sinPhyX * cosPsyZ;

    vec3f column3;
    column3.x = -sinTethaY;
    column3.y = sinPhyX * cosTethaY;
    column3.z = cosPhyX * cosTethaY;

    mOrientation = column1 * mOrientation.x + column2 * mOrientation.y +
        column3 * mOrientation.z;

    mUp = column1 * mUp.x + column2 * mUp.y + column3 * mUp.z;
}


vec3f CameraManager::rotateVector(
                                 const vec3f& aVec,
                                 const vec3f& aAxis,
                                 const float aAngle)
{
    const vec3f c = aVec.dot(aAxis) * aAxis;
    const vec3f u = aVec - c;
    const vec3f v = aAxis.cross(u);

    return c + u * cosf(aAngle) + v * sinf(aAngle);
}

void CameraManager::moveUp(const float aAmount)
{
    mPosition += aAmount * mUp;
}

void CameraManager::moveRight(const float aAmount)
{
    mPosition += aAmount * mRight;
}

void CameraManager::moveForward(const float aAmount)
{
    mPosition += aAmount * mOrientation;
}

void CameraManager::read (const char* aFileName)
{
    std::ifstream input;
    input.open(aFileName);

    if(!input)
        std::cerr << "Could not open file " << aFileName << " for reading.\n"
        << __FILE__ << __LINE__ << std::endl;

    std::string line, buff;
    vec3f hlp;
    float fov;
    int resX, resY;

    while ( !input.eof() )
    {
        std::getline(input, line);

        while(line.find_first_of('#', 0) != std::string::npos)//eat comments
        {
            if(input.eof())
            {
                break;
            }
            std::getline(input, line);
        }

        std::stringstream ss(line);
        ss >> buff;

        if (buff == "position")
        {
            ss >> hlp.x >> hlp.y >> hlp.z;
            setPosition(hlp);
        } 
        else if (buff == "orientation")
        {
            ss >> hlp.x >> hlp.y >> hlp.z;
            setOrientation(hlp);
        } 
        else if (buff == "up")
        {
            ss >> hlp.x >> hlp.y >> hlp.z;
            setUp(hlp);
        }
        else if (buff == "rotation")
        {
            ss >> hlp.x >> hlp.y >> hlp.z;
            setRotation(hlp);
        }
        else if (buff == "resX")
        {
            ss >> resX;
            setResX(resX);
        }
        else if (buff == "resY")
        {
            ss >> resY;
            setResY(resY);
        }
        else if (buff == "FOV")
        {
            ss >> fov;
            setFOV(fov);
        }
    }

    rotate(mRotation);
    mRotation = vec3f::rep(0.f);
    setRight(~(mOrientation % mUp));

    input.close();
}

void CameraManager::write(const char* aFileName) const
{
    std::ofstream cameraFile(aFileName);

    if(!cameraFile)
        std::cerr << "Could not open file " << aFileName << " for writing.\n"
        << __FILE__ << __LINE__ << std::endl;

    cameraFile << "#camera parameters" << std::endl;

    cameraFile << "position\t\t" 
        << getPosition().x << " "
        << getPosition().y << " " 
        << getPosition().z << " " << std::endl;

    cameraFile << "orientation\t\t" 
        << getOrientation().x << " "
        << getOrientation().y << " "
        << getOrientation().z << " " << std::endl;

    cameraFile << "up\t\t\t"
        << getUp().x << " "
        << getUp().y << " "
        << getUp().z << " "
        << std::endl;

    cameraFile << "rotation\t\t"
        << getRotation().x << " "
        << getRotation().y << " "
        << getRotation().z << " " << std::endl;

    cameraFile << "resX\t\t\t" << getResX() << std::endl;
    cameraFile << "resY\t\t\t" << getResY() << std::endl;
    cameraFile << "FOV\t\t\t"  << getFOV()  << std::endl;

    cameraFile.close();
}
