/****************************************************************************/
/* Copyright (c) 2009, Stefan Popov, Javor Kalojanov
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
#include "FWObject.hpp"

#ifndef _WIN32
#include <libgen.h>
#define _strnicmp strncasecmp
//patch for gcc-4.3
#include <stdlib.h>
#include <string.h>
#endif

#ifdef _WIN32
#define _PATH_SEPARATOR '\\'
#else
#define _PATH_SEPARATOR '/'
#endif

typedef std::vector<vec3f> t_vecVector;
typedef std::vector<FWObject::Face> t_faceVector;
typedef std::vector<FWObject::Material> t_materialVector;
typedef std::vector<vec2f> t_texCoordVector;

t_vecVector vertices;
t_vecVector normals;
t_faceVector faces;
t_materialVector materials;
t_texCoordVector texCoords;
std::map<std::string, size_t> materialMap;

namespace objLoaderUtil
{
	typedef std::map<std::string, size_t> t_materialMap;

	void skipWS(const char * &aStr)
	{
		while(isspace(*aStr))
			aStr++;
	}

	std::string endSpaceTrimmed(const char* _str)
	{
		size_t len = strlen(_str);
		const char *firstChar = _str;
		const char *lastChar = firstChar + len - 1;
		while(lastChar >= firstChar && isspace(*lastChar))
			lastChar--;

		return std::string(firstChar, lastChar + 1);
	}


	std::string getDirName(const std::string& _name)
	{
		std::string objDir;
#if _MSC_VER >= 1400
		char fileDir[4096];
		_splitpath_s(_name.c_str(), NULL, 0, fileDir, sizeof(fileDir), NULL, 0, NULL, 0);
		objDir = fileDir;
#endif

#ifndef _WIN32
		char *fnCopy = strdup(_name.c_str());
		const char* dirName = dirname(fnCopy);
		objDir = dirName;
		free(fnCopy);
		//cudastd::log::out << "Dirname: " << objDir << "\n";

#endif // _WIN32

		return objDir;
	}


	void readMtlLib(const std::string &aFileName, t_materialVector &aMatVector, const t_materialMap &aMatMap)
	{
        std::ifstream matInput(aFileName.c_str(), std::ios::in|std::ios::binary);
		std::string buf;

		if(matInput.fail())
			throw std::runtime_error("Error opening .mtl file");

		size_t curMtl = -1, curLine = 0;

		while(!matInput.eof())
		{
			std::getline(matInput, buf);
			curLine++;
			const char* cmd = buf.c_str();
			skipWS(cmd);

			if(_strnicmp(cmd, "newmtl", 6) == 0)
			{
				cmd += 6;

				skipWS(cmd);
				std::string name = endSpaceTrimmed(cmd);
				if(aMatMap.find(name) == aMatMap.end())
					goto parse_err_found;

				curMtl = aMatMap.find(name)->second;
			}
			else if(
				_strnicmp(cmd, "Kd", 2) == 0 || _strnicmp(cmd, "Ks", 2) == 0
				|| _strnicmp(cmd, "Ka", 2) == 0)
			{
				char coeffType = *(cmd + 1);

				if(curMtl == -1)
					goto parse_err_found;

				vec3f color;
				cmd += 2;

				char *newCmdString;
				for(int i = 0; i < 3; i++)
				{
					skipWS(cmd);
					((float*)&color)[i] = (float)strtod(cmd, &newCmdString);
					if(newCmdString == cmd) goto parse_err_found;
					cmd = newCmdString;
				}


				switch (coeffType)
				{
					case 'd':
						aMatVector[curMtl].diffuseCoeff = color; break;
					case 'a':
						aMatVector[curMtl].ambientCoeff = color; break;
					case 's':
						aMatVector[curMtl].specularCoeff = color; break;
				}
			}
			else if(_strnicmp(cmd,  "Ns", 2) == 0)
			{
				if(curMtl == -1)
					goto parse_err_found;

				cmd += 2;

				char *newCmdString;
				skipWS(cmd);
				float coeff = (float)strtod(cmd, &newCmdString);
				if(newCmdString == cmd) goto parse_err_found;
				cmd = newCmdString;
				aMatVector[curMtl].specularExp = coeff;
			}

			continue;
parse_err_found:
			std::cerr << "Error at line " << curLine << "in " << aFileName <<std::endl;
		}
	}
}

using namespace objLoaderUtil;

void FWObject::read(const char* aFileName)
{
    vertices.clear();
    normals.clear();
    faces.clear();
    materials.clear();
    texCoords.clear();
    materialMap.clear();

	std::ifstream inputStream(aFileName, std::ios::in);
	std::string buf;

	const size_t _MAX_BUF = 8192;
	const size_t _MAX_IDX = _MAX_BUF / 2;

	float tmpVert[3];
	size_t tmpIdx[_MAX_IDX * 3];
	int tmpVertPointer, tmpIdxPtr, vertexType;
	size_t curMat = 0, curLine = 0;
	std::vector<std::string> matFiles;

	Material defaultMaterial;
	defaultMaterial.name = "Default";
    defaultMaterial.diffuseCoeff = vec3f::rep(.9774f);
    defaultMaterial.specularCoeff = vec3f::rep(0.f);
	defaultMaterial.specularExp = 0;
    defaultMaterial.ambientCoeff = vec3f::rep(0.2f);
	materials.push_back(defaultMaterial);

	materialMap.insert(std::make_pair(defaultMaterial.name, (size_t)0));

	if(inputStream.fail())
		throw std::runtime_error("Error opening .obj file");

	while(!inputStream.eof())
	{
		std::getline(inputStream, buf);
		const char *cmdString = buf.c_str();

		curLine++;
		skipWS(cmdString);
		switch(tolower(*cmdString))
		{
		case 0:
			break;
		case 'v':
			cmdString++;
			switch(tolower(*cmdString))
			{
				case 'n': vertexType = 1; cmdString++; break;
				case 't': vertexType = 2; cmdString++; break;
				default:
					if(isspace(*cmdString))
						vertexType = 0;
					else
						goto parse_err_found;
			}

			tmpVertPointer = 0;
			for(;;)
			{
				skipWS(cmdString);
				if(*cmdString == 0)
					break;

				char *newCmdString;
				float flt = (float)strtod(cmdString, &newCmdString);
				if(newCmdString == cmdString)
					goto parse_err_found;

				cmdString = newCmdString;

				if(tmpVertPointer >= sizeof(tmpVert) / sizeof(float))
					goto parse_err_found;

				tmpVert[tmpVertPointer++] = flt;
			}

			if(vertexType != 2 && tmpVertPointer != 3 || vertexType == 2 && tmpVertPointer < 2)
				goto parse_err_found;


			if(vertexType == 0)
            {
				vertices.push_back(*(vec3f*)tmpVert);
            }
			else if (vertexType == 1)
            {
				normals.push_back(*(vec3f*)tmpVert);
            }
			else
				texCoords.push_back(*(vec2f*)tmpVert);

			break;

		case 'f':
			cmdString++;
			if(tolower(*cmdString) == 'o')
				cmdString++;
			skipWS(cmdString);

			tmpIdxPtr = 0;
			for(;;)
			{
				if(tmpIdxPtr + 3 >= sizeof(tmpIdx) / sizeof(int))
					goto parse_err_found;

				char *newCmdString;
				int idx = strtol(cmdString, &newCmdString, 10);

				if(cmdString == newCmdString)
					goto parse_err_found;

				cmdString = newCmdString;

				tmpIdx[tmpIdxPtr++] = idx - 1;

				skipWS(cmdString);

				if(*cmdString == '/')
				{
					cmdString++;

					skipWS(cmdString);
					if(*cmdString != '/')
					{
						idx = strtol(cmdString, &newCmdString, 10);

						if(cmdString == newCmdString)
							goto parse_err_found;

						cmdString = newCmdString;

						tmpIdx[tmpIdxPtr++] = idx - 1;
					}
					else
						tmpIdx[tmpIdxPtr++] = -1;


					skipWS(cmdString);
					if(*cmdString == '/')
					{
						cmdString++;
						skipWS(cmdString);
						idx = strtol(cmdString, &newCmdString, 10);

						//Do ahead lookup of one number
						skipWS((const char * &)newCmdString);
						if(isdigit(*newCmdString) || (*newCmdString == 0 || *newCmdString == '#') && cmdString != newCmdString)
						{
							if(cmdString == newCmdString)
								goto parse_err_found;

							cmdString = newCmdString;

							tmpIdx[tmpIdxPtr++] = idx - 1;
						}
						else
							tmpIdx[tmpIdxPtr++] = -1;
					}
					else
						tmpIdx[tmpIdxPtr++] = -1;
				}
				else
				{
					tmpIdx[tmpIdxPtr++] = -1;
					tmpIdx[tmpIdxPtr++] = -1;
				}

				skipWS(cmdString);
				if(*cmdString == 0)
					break;
			}

			if(tmpIdxPtr <= 6)
				goto parse_err_found;

			for(int idx = 3; idx < tmpIdxPtr - 3; idx += 3)
			{
				Face t(this);
				t.material = curMat;
				memcpy(&t.vert1, tmpIdx, 3 * sizeof(size_t));
				memcpy(&t.vert2, tmpIdx + idx, 6 * sizeof(size_t));

				faces.push_back(t);
			}
			break;

		case 'o':
		case 'g':
		case 's': //?
		case '#':
			//Not supported
			break;

		default:
			if(_strnicmp(cmdString, "usemtl", 6) == 0)
			{
				cmdString += 6;
				skipWS(cmdString);
				std::string name = endSpaceTrimmed(cmdString);
				if(name.empty())
					goto parse_err_found;

				if(materialMap.find(name) == materialMap.end())
				{
					materials.push_back(Material(name.c_str()));
					materialMap[name] = materials.size() - 1;
				}

				curMat = materialMap[name];
			}
			else if(_strnicmp(cmdString, "mtllib", 6) == 0)
			{
				cmdString += 6;
				skipWS(cmdString);
				std::string name = endSpaceTrimmed(cmdString);
				if(name.empty())
					goto parse_err_found;

				matFiles.push_back(name);
			}
			else
			{
				std::cerr << "Unknown entity at line " << curLine << std::endl;
			}
		}

		continue;
parse_err_found:
		std::cerr << "Error at line " << curLine << std::endl;
	}


	std::string objDir = getDirName(aFileName);

	for(std::vector<std::string>::const_iterator it = matFiles.begin(); it != matFiles.end(); it++)
	{
		std::string mtlFileName = objDir + _PATH_SEPARATOR + *it;

		readMtlLib(mtlFileName, materials, materialMap);
	}

    //normalize reflectance coefficients
    for(t_materialVector::iterator it = materials.begin(); it != materials.end();
        ++it)
    {
        it->setupPhongCoefficients();
    }


	for(t_faceVector::iterator it = faces.begin(); it != faces.end(); it++)
	{
		if(it->norm1 == -1 || it->norm2 == -1 || it->norm3 == -1)
		{
			vec3f e1 = vertices[it->vert2] - vertices[it->vert1];
			vec3f e2 = vertices[it->vert3] - vertices[it->vert1];
			vec3f n = ~(e1 % e2);
			if(it->norm1 == -1) it->norm1 = normals.size();
			if(it->norm2 == -1) it->norm2 = normals.size();
			if(it->norm3 == -1) it->norm3 = normals.size();

			normals.push_back(n);
		}
	}

    //////////////////////////////////////////////////////////////////////////
    //Copy to dynamically allocated arrays...
    //NVCC-std::vector-bug workaround
    //////////////////////////////////////////////////////////////////////////

    if (vertices.size() > 0u)
    {
        allocateVertices(vertices.size());

        for (size_t it = 0; it < vertices.size(); ++it)
        {
            mVertices[it] = vertices[it];
        }
    }

    if(normals.size() > 0u)
    {

        allocateNormals(normals.size());

        for (size_t it = 0; it < normals.size(); ++it)
        {
            mNormals[it] = normals[it];
        }
    }

    if (faces.size() > 0u)
    {
        allocateFaces(faces.size());

        for (size_t it = 0; it < faces.size(); ++it)
        {
            mFaces[it] = faces[it];
        }
    }

    if (materials.size() > 0u)
    {
        allocateMaterials(materials.size());

        for (size_t it = 0; it < materials.size(); ++it)
        {
            mMaterials[it] = materials[it];
        }
    }

    if (texCoords.size() > 0u)
    {
        allocateTexCoords(texCoords.size());

        for (size_t it = 0; it < texCoords.size(); ++it)
        {
            mTexCoords[it] = texCoords[it];
        }
    }
}
