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

#ifdef _MSC_VER
#pragma once
#endif

#ifndef FWOBJECT_HPP_INCLUDED_45834E1E_1D79_4F3E_ABD3_77E318EF9223
#define FWOBJECT_HPP_INCLUDED_45834E1E_1D79_4F3E_ABD3_77E318EF9223

#include "../Core/Algebra.hpp"

//A Wavefront3D object. It provides the functionality of 
//	an indexed triangle mesh. The primitive consists
//	of 5 buffers holding vertex positions, vertex normals,
//	vertex texture coordinates, face materials, and the faces
//	them selfs. Each face is a triangle and only contains indices
//	into the position, normal, texture coordinate, and material
//	buffers.
class FWObject
{
public:
    class Face;

public:
    FWObject():mVertices(NULL), mNormals(NULL), mFaces(NULL),
        mMaterials(NULL), mTexCoords(NULL)
    {}

    ~FWObject()
    {
        if (mVertices != NULL)
        {
            delete[] mVertices;
        }
        if (mNormals != NULL)
        {
            delete[] mNormals;
        }
        if (mFaces != NULL)
        {
            delete[] mFaces;
        }
        if (mMaterials != NULL)
        {
            delete[] mMaterials;
        }
        if (mTexCoords != NULL)
        {
            delete[] mTexCoords;
        }
    }

    //Represents a material
    struct Material
    {
        //The unique name of the material
        const char* name;
        //The diffuse color of the material
        vec3f diffuseCoeff;

        vec3f specularCoeff;

        vec3f ambientCoeff;

        float specularExp;

        Material():ambientCoeff(vec3f::rep(0.f)),
            diffuseCoeff(vec3f::rep(0.f)),
            specularCoeff(vec3f::rep(0.f)),
            specularExp(0.f)
        {}
        Material(const char* _name) : name(_name),
            ambientCoeff(vec3f::rep(0.f)),
            diffuseCoeff(vec3f::rep(0.f)),
            specularCoeff(vec3f::rep(0.f)),
            specularExp(0.f)
        {}

        void setupPhongCoefficients()
        {
            diffuseCoeff  = diffuseCoeff / static_cast<float>(M_PI);
            specularCoeff = specularCoeff * (specularExp + 2) * 0.5f / static_cast<float>(M_PI);
        }

    };


    //A face definition (triangle)
    class Face
    {
        FWObject *m_lwObject;
    public:
        size_t material;
        size_t vert1, tex1, norm1;
        size_t vert2, tex2, norm2;
        size_t vert3, tex3, norm3;

        Face() : m_lwObject(NULL)
        {}

        Face(FWObject * _obj) : m_lwObject(_obj) {}
    };


    size_t getNumVertices() const
    {
        return mNumVertices;
    }
    size_t getNumNormals() const
    {
        return mNumNormals;
    }
    size_t getNumFaces() const
    {
        return mNumFaces;
    }
    size_t getNumMaterials() const
    {
        return mNumMaterials;
    }
    size_t getNumTexCoords() const
    {
        return mNumTexCoords;
    }
    
    vec3f getVertex(size_t aVtxId) const
    {
        return mVertices[aVtxId];
    }

    vec3f getNormal(size_t aNormalId) const
    {
        return mNormals[aNormalId];
    }

    Face getFace(size_t aFaceId) const
    {
        return mFaces[aFaceId];
    }

    Material getMaterial(size_t aMatId) const
    {
        return mMaterials[aMatId];
    }

    vec2f getTexCoords(size_t aCoordId) const
    {
        return mTexCoords[aCoordId];
    }

    vec3f& getVertex(size_t aVtxId)
    {
        return mVertices[aVtxId];
    }

    vec3f& getNormal(size_t aNormalId)
    {
        return mNormals[aNormalId];
    }

    Face& getFace(size_t aFaceId)
    {
        return mFaces[aFaceId];
    }

    Material& getMaterial(size_t aMatId)
    {
        return mMaterials[aMatId];
    }

    vec2f& getTexCoords(size_t aCoordId)
    {
        return mTexCoords[aCoordId];
    }

    void allocateVertices(const size_t aSize)
    {
        if (mVertices != NULL)
        {
            delete[] mVertices;
        }

        mVertices = new vec3f[aSize];
        mNumVertices = aSize;
    }


    void allocateNormals(const size_t aSize)
    {
        if (mNormals != NULL)
        {
            delete[] mNormals;
        }

        mNormals = new vec3f[aSize];
        mNumNormals = aSize;
    }

    void allocateFaces(const size_t aSize)
    {
        if (mFaces != NULL)
        {
            delete[] mFaces;
        }

        mFaces = new Face[aSize];
        mNumFaces = aSize;

    }

    void allocateMaterials(const size_t aSize)
    {
        if (mMaterials != NULL)
        {
            delete[] mMaterials;
        }

        mMaterials = new Material[aSize];
        mNumMaterials = aSize;

    }

    void allocateTexCoords(const size_t aSize)
    {
        if (mTexCoords != NULL)
        {
            delete[] mTexCoords;
        }

        mTexCoords = new vec2f[aSize];
        mNumTexCoords = aSize;

    }

    //Reads the FrontWave3D object from a file
    void read(const char* aFileName);

    typedef const vec3f* t_VertexIterator;
    typedef const Face* t_FaceIterator;
    typedef const Material* t_MaterialIterator;

    t_VertexIterator verticesBegin() const
    {
        return mVertices;
    }

    t_VertexIterator verticesEnd() const
    {
        return mVertices + mNumVertices;
    }

    t_FaceIterator facesBegin() const
    {
        return mFaces;
    }

    t_FaceIterator facesEnd() const
    {
        return mFaces + mNumFaces;
    }

    t_MaterialIterator materialsBegin() const
    {
        return mMaterials;
    }

    t_MaterialIterator materialsEnd() const
    {
        return mMaterials + mNumMaterials;
    }

private:
    size_t mNumVertices, mNumNormals, mNumFaces, mNumMaterials, mNumTexCoords;
    vec3f* mVertices;
    vec3f* mNormals;
    Face*  mFaces;
    Material* mMaterials;
    vec2f* mTexCoords;
};

#endif // FWOBJECT_HPP_INCLUDED_45834E1E_1D79_4F3E_ABD3_77E318EF9223
